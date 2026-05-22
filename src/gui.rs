//! GPU-accelerated GUI for GSEA using egui + eframe (wgpu backend).
//! Non-blocking UI with waiting indicator and model switcher.

use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use eframe::egui::{self, Color32, CornerRadius, Vec2, FontFamily, FontId};

use crate::agent::Agent;
use crate::memory_brain::Brain;
use crate::tools::ToolRegistry;

#[derive(Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

enum GuiEvent {
    StreamChunk(String),
    StreamDone,
    Error(String),
    StatsUpdate(String, usize), // (brain_stats_json, tool_count)
}

/// Dashboard data displayed in the sidebar.
#[derive(Default, Clone)]
struct DashboardData {
    workflow_count: u32,
    last_workflow: String,
    agent_context_sizes: Vec<(String, usize)>,
    total_messages_processed: u64,
}

pub struct GseaGui {
    agent: Arc<Mutex<Option<Agent>>>,
    brain: Arc<Mutex<Brain>>,
    registry: Arc<Mutex<ToolRegistry>>,
    messages: Vec<ChatMessage>,
    input_buf: String,
    brain_stats: String,
    tool_count: usize,
    model_name: String,
    is_processing: bool,
    rx: mpsc::Receiver<GuiEvent>,
    tx: mpsc::Sender<GuiEvent>,
    dashboard: DashboardData,
    uptime: std::time::Instant,
}

impl GseaGui {
    pub fn new(
        agent: Option<Agent>,
        brain: Arc<Mutex<Brain>>,
        registry: Arc<Mutex<ToolRegistry>>,
        model_name: &str,
    ) -> Self {
        let stats = {
            let b = brain.lock().unwrap();
            serde_json::to_string_pretty(&b.stats()).unwrap_or_default()
        };
        let tc = registry.lock().unwrap().list_tools().len();
        let (tx, rx) = mpsc::channel();

        Self {
            agent: Arc::new(Mutex::new(agent)),
            brain, registry,
            messages: Vec::new(),
            input_buf: String::new(),
            brain_stats: stats,
            tool_count: tc,
            model_name: model_name.to_string(),
            is_processing: false,
            rx, tx,
            dashboard: DashboardData::default(),
            uptime: std::time::Instant::now(),
        }
    }

    fn send_message(&mut self, input: &str) {
        let input = input.to_string();
        self.messages.push(ChatMessage { role: "You".into(), content: input.clone() });
        self.is_processing = true;

        let agent = self.agent.clone();
        let brain = self.brain.clone();
        let registry = self.registry.clone();
        let tx = self.tx.clone();

        thread::spawn(move || {
            let tokio_rt = tokio::runtime::Runtime::new();
            match tokio_rt {
                Ok(rt) => {
                    let mut ag = agent.lock().unwrap();
                    if let Some(ref mut a) = *ag {
                        match rt.block_on(a.process_message_stream(&input)) {
                            Ok(mut rx) => {
                                // Forward streaming chunks to GUI
                                rt.block_on(async {
                                    while let Some(chunk) = rx.recv().await {
                                        let _ = tx.send(GuiEvent::StreamChunk(chunk));
                                    }
                                });
                                let _ = tx.send(GuiEvent::StreamDone);
                            }
                            Err(e) => {
                                let _ = tx.send(GuiEvent::Error(e.to_string()));
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(GuiEvent::Error(format!("Runtime error: {}", e)));
                }
            }

            // Update stats
            let b = brain.lock().unwrap();
            let stats = serde_json::to_string_pretty(&b.stats()).unwrap_or_default();
            let tc = registry.lock().unwrap().list_tools().len();
            let _ = tx.send(GuiEvent::StatsUpdate(stats, tc));
        });
    }
}

fn setup_korean_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    if let Ok(data) = std::fs::read("/System/Library/Fonts/AppleSDGothicNeo.ttc") {
        fonts.font_data.insert("korean".to_string(), std::sync::Arc::new(egui::FontData::from_owned(data)));
        fonts.families.get_mut(&FontFamily::Proportional).unwrap().insert(0, "korean".to_string());
    }
    ctx.set_fonts(fonts);
}

impl eframe::App for GseaGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        setup_korean_font(ctx);

        // Drain pending events from the channel
        while let Ok(event) = self.rx.try_recv() {
            match event {
                GuiEvent::StreamChunk(chunk) => {
                    // Append to the last GSEA message or create a new one
                    if let Some(last) = self.messages.last_mut() {
                        if last.role == "GSEA" && self.is_processing {
                            last.content.push_str(&chunk);
                        } else {
                            self.messages.push(ChatMessage { role: "GSEA".into(), content: chunk });
                        }
                    } else {
                        self.messages.push(ChatMessage { role: "GSEA".into(), content: chunk });
                    }
                }
                GuiEvent::StreamDone => {
                    self.is_processing = false;
                }
                GuiEvent::Error(text) => {
                    self.messages.push(ChatMessage { role: "Error".into(), content: text });
                    self.is_processing = false;
                }
                GuiEvent::StatsUpdate(stats, tc) => {
                    self.brain_stats = stats;
                    self.tool_count = tc;
                }
            }
        }

        // Header with model switcher
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("⚡ GSEA").color(Color32::from_rgb(100, 180, 255)));
                ui.label(egui::RichText::new("Gemma Self-Evolving Agent").color(Color32::GRAY).size(13.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let models = ["gemma4:26b", "qwen3:8b", "auto"];
                    let mut current_idx = match self.model_name.as_str() {
                        "gemma4:26b" => 0,
                        "qwen3:8b" => 1,
                        _ => 2,
                    };
                    egui::ComboBox::from_id_salt("model_selector")
                        .selected_text(&self.model_name)
                        .width(140.0)
                        .show_ui(ui, |ui| {
                            for (i, m) in models.iter().enumerate() {
                                if ui.selectable_value(&mut current_idx, i, *m).clicked() {
                                    self.model_name = m.to_string();
                                    if let Ok(mut ag) = self.agent.lock() {
                                        if let Some(ref mut agent) = *ag {
                                            match *m {
                                                "gemma4:26b" => { agent.llm().set_model("gemma4:26b"); agent.fast_llm().set_model("qwen3:8b"); }
                                                "qwen3:8b" => { agent.llm().set_model("qwen3:8b"); agent.fast_llm().set_model("qwen3:8b"); }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        });
                });
            });
        });

        // Input bar
        egui::TopBottomPanel::bottom("input").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let resp = ui.add_sized(
                    ui.available_size() - Vec2::new(90.0, 0.0),
                    egui::TextEdit::singleline(&mut self.input_buf)
                        .hint_text(if self.is_processing { "⏳ 처리 중..." } else { "메시지를 입력하세요..." })
                        .font(FontId::new(14.0, FontFamily::Proportional))
                        .desired_width(f32::INFINITY),
                );

                let btn_label = if self.is_processing { "⏳" } else { "⏎ 전송" };
                let btn = egui::Button::new(egui::RichText::new(btn_label).size(14.0).color(Color32::WHITE))
                    .fill(if self.is_processing { Color32::from_rgb(80, 80, 80) } else { Color32::from_rgb(50, 120, 220) })
                    .min_size(Vec2::new(80.0, 28.0));

                let should_send = !self.is_processing && !self.input_buf.trim().is_empty()
                    && (ui.add(btn).clicked() || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))));

                if should_send {
                    let input = self.input_buf.trim().to_string();
                    self.input_buf.clear();
                    self.send_message(&input);
                }
                if !resp.has_focus() && !self.is_processing { resp.request_focus(); }
            });
        });

        // Status bar
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let status = if self.is_processing {
                    format!("⏳ waiting... | 💬 {} | 🔧 {} tools | {}", self.messages.len(), self.tool_count, self.model_name)
                } else {
                    format!("💬 {} | 🔧 {} tools | {}", self.messages.len(), self.tool_count, self.model_name)
                };
                ui.label(egui::RichText::new(status).size(11.0).color(Color32::GRAY));
            });
        });

        // Update dashboard data periodically
        {
            if let Ok(ag) = self.agent.lock() {
                if let Some(ref a) = *ag {
                    self.dashboard.total_messages_processed = a.total_messages;
                }
            }
        }

        // Sidebar
        egui::SidePanel::right("sidebar").resizable(true).default_width(240.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // ── Dashboard ──
                ui.label(egui::RichText::new("📊 Dashboard").size(14.0).strong());
                ui.separator();
                let elapsed = self.uptime.elapsed();
                let mins = elapsed.as_secs() / 60;
                let secs = elapsed.as_secs() % 60;
                ui.label(egui::RichText::new(format!("Uptime: {}m {}s", mins, secs)).size(11.0).color(Color32::LIGHT_GRAY));
                ui.label(egui::RichText::new(format!("Messages: {}", self.dashboard.total_messages_processed)).size(11.0).color(Color32::LIGHT_GRAY));
                ui.label(egui::RichText::new(format!("Chat history: {}", self.messages.len())).size(11.0).color(Color32::LIGHT_GRAY));
                if self.is_processing {
                    ui.label(egui::RichText::new("Status: Processing…").size(11.0).color(Color32::from_rgb(255, 200, 50)));
                } else {
                    ui.label(egui::RichText::new("Status: Ready").size(11.0).color(Color32::from_rgb(80, 220, 150)));
                }
                ui.add_space(12.0);

                // ── Brain ──
                ui.label(egui::RichText::new("🧠 Brain").size(14.0).strong());
                ui.separator();
                ui.monospace(egui::RichText::new(&self.brain_stats).size(11.0).color(Color32::LIGHT_GRAY));
                ui.add_space(12.0);

                // ── Tools ──
                ui.label(egui::RichText::new("🔧 Tools").size(14.0).strong());
                ui.separator();
                let tools = self.registry.lock().unwrap();
                for t in tools.list_tools() {
                    ui.label(egui::RichText::new(format!("• {}", t.name())).size(12.0).color(Color32::from_rgb(180, 190, 200)));
                }
                drop(tools);
                ui.add_space(12.0);

                // ── Skills ──
                ui.label(egui::RichText::new("💡 Skills").size(14.0).strong());
                ui.separator();
                let b = self.brain.lock().unwrap();
                let skills = b.list_skills();
                if skills.is_empty() {
                    ui.label(egui::RichText::new("(none)").color(Color32::GRAY).size(12.0));
                } else {
                    for (name, desc) in &skills {
                        ui.label(egui::RichText::new(format!("• {}: {}", name, desc)).size(11.0).color(Color32::from_rgb(160, 200, 160)));
                    }
                }
                drop(b);
                ui.add_space(12.0);

                // ── Recent Memories ──
                ui.label(egui::RichText::new("🔖 Recent Memories").size(14.0).strong());
                ui.separator();
                let b = self.brain.lock().unwrap();
                let recent = b.recall("", 5);
                if recent.is_empty() {
                    ui.label(egui::RichText::new("(empty)").color(Color32::GRAY).size(11.0));
                } else {
                    for item in &recent {
                        let preview: String = item.content.chars().take(60).collect();
                        ui.label(egui::RichText::new(format!("[{}] {}", item.memory_type, preview)).size(10.0).color(Color32::from_rgb(160, 170, 190)));
                    }
                }
            });
        });

        // Chat area with waiting overlay
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                // Waiting indicator at top when processing
                if self.is_processing {
                    ui.horizontal(|ui| {
                        ui.add_space(10.0);
                        ui.label(egui::RichText::new("⏳ waiting...").color(Color32::from_rgb(255, 200, 50)).size(16.0).strong());
                        ui.label(egui::RichText::new("모델이 응답 중입니다").color(Color32::GRAY).size(12.0));
                    });
                    ui.add_space(8.0);
                }

                for msg in &self.messages {
                    let is_user = msg.role == "You";
                    let bg = if is_user { Color32::from_rgb(50, 80, 140) } else if msg.role == "Error" { Color32::from_rgb(120, 40, 40) } else { Color32::from_rgb(30, 34, 42) };
                    let name_c = if is_user { Color32::from_rgb(100, 180, 255) } else { Color32::from_rgb(80, 220, 150) };
                    ui.add_space(6.0);
                    egui::Frame::new().fill(bg).corner_radius(CornerRadius::same(10)).inner_margin(egui::Margin::symmetric(14, 10)).show(ui, |ui| {
                        ui.label(egui::RichText::new(&msg.role).color(name_c).size(12.0).strong());
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(&msg.content).color(if is_user { Color32::WHITE } else { Color32::from_rgb(210, 215, 225) }).size(13.0));
                    });
                }
                ui.add_space(20.0);
            });
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }
}
