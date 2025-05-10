import networkx as nx

ori_path = "datasets_kg/hotel.graphml"

def load_knowledge_graph(path=ori_path):
    return nx.read_graphml(path)

def build_term_to_aspect_map(graph):
    term2aspect = {}
    for source, target, data in graph.edges(data=True):
        if data.get("relation") == "has-aspect":
            term2aspect[source] = target
    return term2aspect

def map_terms_to_aspects(terms, term2aspect_map):
    return [(term, term2aspect_map.get(term, "UNKNOWN")) for term in terms]

def main():
    graph = load_knowledge_graph(ori_path)
    term2aspect_map = build_term_to_aspect_map(graph)
    
    terms = ["thang máy", "nhân viên", "đồ ăn", "giá cả"]
    mappings = map_terms_to_aspects(terms, term2aspect_map)

    for term, aspect in mappings:
        print(f"{term} → {aspect}")

if __name__ == "__main__":
    main()
