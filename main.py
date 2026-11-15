# my own samll rag implementation
import basic_rag
import prod_hybrid_search
import langsmith_setup
import semantic_chunking

def main():

    print("Entry POINT STARTED")
    #basic_rag.basic_rag()
    #prod_hybrid_search.prod_hybrid_search()
    #langsmith_setup.demo_basic_tracing()

    semantic_chunking.semantic_vs_recursive_chunking()

if __name__ == "__main__":
    main()

