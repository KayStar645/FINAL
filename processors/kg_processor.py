import networkx as nx
import os
import json
import networkx as nx
import matplotlib.pyplot as plt

class SemanticGraphBuilder:
    def __init__(self, jsonl_data):
        self.G = nx.DiGraph()
        self._add_metadata_keys()

        for item in jsonl_data:
            text = item.get("data", "")
            for span in item.get("label", []):
                if len(span) != 3:
                    continue
                start, end, full_label = span
                term = text[start:end].strip().lower()
                try:
                    domain, aspect, sentiment = full_label.split("#")
                except ValueError:
                    continue

                aspect_full = f"{domain}#{aspect}"

                # Tạo các node
                self._add_node(f"term::{term}", "term")
                self._add_node(f"aspect::{aspect_full}", "aspect")
                self._add_node(f"domain::{domain}", "domain")
                self._add_node(f"sentiment::{sentiment.lower()}", "sentiment")

                # Tạo các cạnh
                self._add_edge(f"term::{term}", f"aspect::{aspect_full}", "has-aspect")
                self._add_edge(f"aspect::{aspect_full}", f"domain::{domain}", "belongs-to")
                self._add_edge(f"aspect::{aspect_full}", f"sentiment::{sentiment.lower()}", "expresses")

    def _add_metadata_keys(self):
        # GraphML metadata keys
        self.G.graph['node_default'] = {'type': ''}
        self.G.graph['edge_default'] = {'relation': ''}

    def _add_node(self, node_id, node_type):
        self.G.add_node(node_id, type=node_type)

    def _add_edge(self, source, target, relation):
        self.G.add_edge(source, target, relation=relation)

    def export(self, path: str):
        nx.write_graphml(self.G, path)

def create_semantic_graphml(jsonl_path: str, output_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    graph = SemanticGraphBuilder(data)
    graph.export(output_path)

def show():
    G = nx.read_graphml("datasets/hotel_semantic.graphml")
    pos = nx.spring_layout(G, k=0.5)
    node_types = nx.get_node_attributes(G, 'type')

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8)
    plt.title("Semantic Graph")
    plt.show()

if __name__ == "__main__":
    input_file = "datasets/hotel.jsonl"  # <-- sửa đường dẫn file của bạn ở đây
    output_file = "datasets/hotel_semantic.graphml"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    create_semantic_graphml(input_file, output_file)
    print(f"✅ Đã tạo file: {output_file}")
    #show()
