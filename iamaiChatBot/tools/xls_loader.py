from langchain_community.document_loaders import UnstructuredExcelLoader

loader = UnstructuredExcelLoader("data/stanley-cups.xlsx", mode="elements")
docs = loader.load()
docs[0]


