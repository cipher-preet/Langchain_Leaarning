from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',          # ---------- Name ypur curreent Directory ----------------
    glob='*.pdf',
    loader_cls=PyPDFLoader   
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)