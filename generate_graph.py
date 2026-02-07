from src.core.engine import create_legal_engine

def save_graph_image(output_path="graph.png"):
    engine = create_legal_engine()
    try:
        # Generate the PNG data
        graph_png = engine.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(graph_png)
            
        print(f"Graph successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error generating graph: {e}")
        print("Tip: Ensure 'pygraphviz' or 'mermaid' is installed.")

if __name__ == "__main__":
    save_graph_image()
