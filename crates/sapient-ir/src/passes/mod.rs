//! IR optimisation passes.
//!
//! Each pass implements the `Pass` trait and transforms the graph in-place.

pub mod constant_folding;
pub mod dead_code;
pub mod layout;
pub mod fusion;

use sapient_core::error::Result;
use crate::graph::Graph;

// ── Pass trait ────────────────────────────────────────────────────────────────

/// A graph-transformation pass.
pub trait Pass: std::fmt::Debug {
    /// Human-readable name of the pass.
    fn name(&self) -> &str;

    /// Run the pass, mutating the graph in-place.
    fn run(&self, graph: &mut Graph) -> Result<()>;
}

// ── PassManager ──────────────────────────────────────────────────────────────

/// Runs a sequence of passes over a graph, reporting changes.
#[derive(Debug, Default)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pass.
    pub fn add<P: Pass + 'static>(&mut self, pass: P) -> &mut Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Run all passes in order.
    pub fn run_all(&self, graph: &mut Graph) -> Result<()> {
        for pass in &self.passes {
            tracing::debug!(pass = pass.name(), "running IR pass");
            pass.run(graph)?;
        }
        Ok(())
    }

    /// Build the standard optimisation pipeline.
    pub fn standard() -> Self {
        let mut pm = Self::new();
        pm.add(constant_folding::ConstantFoldingPass)
          .add(dead_code::DeadCodeEliminationPass)
          .add(layout::LayoutOptimizationPass)
          .add(fusion::OperatorFusionPass);
        pm
    }
}
