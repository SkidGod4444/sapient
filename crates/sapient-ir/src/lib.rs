//! SAPIENT Intermediate Representation — compute graph, ops, and passes.

pub mod graph;
pub mod node;
pub mod op;
pub mod passes;
pub mod shape_inference;

pub use graph::{Edge, Graph};
pub use node::{Node, NodeId};
pub use op::OpType;
pub use passes::Pass;
pub use shape_inference::ShapeRegistry;
