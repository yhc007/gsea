//! GPU-accelerated GUI for GSEA using egui + eframe (wgpu backend).
//! DeepSeek TUI-style split panel layout.

use std::sync::Arc;

use eframe::egui;

use crate::agent::Agent;
use crate::memory_brain::{Brain, MemoryType};
use crate::tools::ToolRegistry;

/// Chat message display
#[derive(Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

/// GSEA GUI application
pub struct GseaGui {
    agent: Option<Agent>,
    brain: Arc<std::sync::Mutex<Brain>>,
    registry: Arc<std::sync::Mutex<ToolRegistry>>,

    messages: Vec<ChatMessage>,
    input_buf: String,

    // Cached state
    brain_stats: String,
    tool_count: usize,
    model_name: String,

    // Scroll state
    auto_scroll: bool,
}

impl GseaGui {
    pub fn new(
        agent: Option<Agent>,
        brain: Arc<std::sync::Mutex<Brain>>,
        registry: Arc<std::sync::Mutex<ToolRegistry>>,
        model_name: &str,
    ) -> Self {
        let stats = {
            let b = brain.lock().unwrap();
            serde_json::to_string_pretty(&b.stats()).unwrap_or_default()
        };
        let tc = registry.lock().unwrap().list_tools().len();

        Self {
            agent,
            brain,
            registry,
            messages: Vec::new(),
            input_buf: String::new(),
            brain_stats: stats,
            tool_count: tc,
            model_name: model_name.to_string(),
            auto_scroll: true,
        }
    }

    fn add_message(&mut self, role: &str, content: &str) {
        let msg = ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        };
        self.messages.push(msg);
        self.auto_scroll = true;
    }

    fn refresh_stats(&mut self) {
        let b = self.brain.lock().unwrap();
        self.brain_stats = serde_json::to_string_pretty(&b.stats()).unwrap_or_default();
        self.tool_count = self.registry.lock().unwrap().list_tools().len();
    }
}

impl eframe::App for GseaGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ─── Top panel: title ────────────────────────────────────
        egui::TopBottomPanel::top("title_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("⚡ GSEA");
                ui.label("Gemma Self-Evolving Agent");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&self.model_name);
                });
            });
        });

        // ─── Bottom panel: input ─────────────────────────────────
        egui::TopBottomPanel::bottom("input_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let resp = ui.add_sized(
                    ui.available_size() - egui::vec2(80.0, 0.0),
                    egui::TextEdit::singleline(&mut self.input_buf)
                        .hint_text("Type a message...")
                        .desired_width(f32::INFINITY),
                );

                let send_clicked = ui.button("⏎ Send").clicked()
                    || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));

                if send_clicked && !self.input_buf.is_empty() {
                    let input = self.input_buf.trim().to_string();
                    self.input_buf.clear();
                    self.add_message("You", &input);

                    // Process via agent (synchronously for now — will spawn task later)
                    if let Some(ref mut agent) = self.agent {
                        let fut = agent.process_message(&input);
                        match futures::executor::block_on(fut) {
                            Ok(response) => {
                                self.add_message("GSEA", &response);
                            }
                            Err(e) => {
                                self.add_message("Error", &e.to_string());
                            }
                        }
                    } else {
                        self.add_message("GSEA", "Agent not initialized");
                    }

                    self.refresh_stats();
                    resp.request_focus();
                }

                if !resp.has_focus() {
                    resp.request_focus();
                }
            });
        });

        // ─── Status bar ──────────────────────────────────────────
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("💬 {} messages", self.messages.len()));
                ui.separator();
                ui.label(format!("🧠 memory"));
                ui.separator();
                ui.label(format!("🔧 {} tools", self.tool_count));
                ui.separator();
                ui.label(format!("📡 {}", self.model_name));
            });
        });

        // ─── Right panel: info sidebar ───────────────────────────
        egui::SidePanel::right("info_panel")
            .resizable(true)
            .default_width(220.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("🧠 Brain");
                    ui.separator();
                    ui.monospace(&self.brain_stats);
                    ui.add_space(10.0);

                    ui.heading("🔧 Tools");
                    ui.separator();
                    let tools = self.registry.lock().unwrap();
                    for t in tools.list_tools() {
                        ui.label(t.name());
                    }
                    ui.add_space(10.0);

                    ui.heading("💡 Skills");
                    ui.separator();
                    let b = self.brain.lock().unwrap();
                    let skills = b.list_skills();
                    if skills.is_empty() {
                        ui.label("(none)");
                    } else {
                        for (name, desc) in &skills {
                            ui.label(format!("• {}: {}", name, desc));
                        }
                    }
                });
            });

        // ─── Central panel: chat history ─────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .max_height(f32::INFINITY)
                .show(ui, |ui| {
                    for msg in &self.messages {
                        let color = match msg.role.as_str() {
                            "You" => egui::Color32::LIGHT_BLUE,
                            "Error" => egui::Color32::LIGHT_RED,
                            _ => egui::Color32::WHITE,
                        };
                        ui.colored_label(color, format!("{}:", msg.role));
                        ui.label(&msg.content);
                        ui.separator();
                    }
                });
        });

        // Repaint continuously for smooth interactivity
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }
}
