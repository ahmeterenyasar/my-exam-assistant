import argparse
import json
import shutil
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COURSES_FILE = CHROMA_DIR / "courses.json"


def discover_pdf_files(data_dir: Path) -> list[Path]:
	return sorted(data_dir.rglob("*.pdf"))


def infer_course_name(pdf_path: Path, data_dir: Path) -> str:
	relative = pdf_path.relative_to(data_dir)
	if len(relative.parts) >= 2:
		return relative.parts[0]
	return "genel"


def load_documents(pdf_files: list[Path], data_dir: Path):
	docs = []
	courses = set()

	for pdf_path in pdf_files:
		course = infer_course_name(pdf_path, data_dir)
		courses.add(course)
		loader = PyPDFLoader(str(pdf_path))
		pdf_docs = loader.load()

		for doc in pdf_docs:
			doc.metadata["course"] = course
			doc.metadata["file_name"] = pdf_path.name
			doc.metadata["source"] = str(pdf_path)
			docs.append(doc)

	return docs, sorted(courses)


def build_vectorstore(reset: bool = False):
	if reset and CHROMA_DIR.exists():
		shutil.rmtree(CHROMA_DIR)

	CHROMA_DIR.mkdir(parents=True, exist_ok=True)

	pdf_files = discover_pdf_files(DATA_DIR)
	if not pdf_files:
		print("PDF bulunamadi. Lutfen data/<ders_adi>/ klasorlerine PDF dosyalari ekleyin.")
		return

	docs, courses = load_documents(pdf_files, DATA_DIR)
	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	split_docs = splitter.split_documents(docs)

	embeddings = OllamaEmbeddings(model="nomic-embed-text")
	vectorstore = Chroma(
		collection_name="ders_notlari",
		persist_directory=str(CHROMA_DIR),
		embedding_function=embeddings,
	)

	vectorstore.add_documents(split_docs)

	COURSES_FILE.write_text(
		json.dumps(
			{
				"courses": courses,
				"total_pdfs": len(pdf_files),
				"total_chunks": len(split_docs),
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)

	print("Ingest tamamlandi.")
	print(f"Toplam PDF: {len(pdf_files)}")
	print(f"Toplam chunk: {len(split_docs)}")
	print("Bulunan dersler:", ", ".join(courses))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="PDF dosyalarini Chroma vektor veritabanina aktarir.")
	parser.add_argument(
		"--reset",
		action="store_true",
		help="Var olan chroma_db klasorunu siler ve sifirdan ingest eder.",
	)
	args = parser.parse_args()

	build_vectorstore(reset=args.reset)
