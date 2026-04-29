from datasets import load_dataset  # this already works on your Nano

flores = load_dataset("your_existing_flores_load_method")  
# OR however you loaded it in your evaluation script — copy that exact line

sources    = [...]  # your 100 English sentences
references = [...]  # your 100 Spanish references
baseline   = [...]  # your 100 greedy translations
mbr        = [...]  # your 100 MBR translations  
qe_rerank  = [...]  # your 100 QE-Rerank translations

with open("sources.txt",   "w") as f: f.write("\n".join(sources))
with open("references.txt","w") as f: f.write("\n".join(references))
with open("baseline.txt",  "w") as f: f.write("\n".join(baseline))
with open("mbr.txt",       "w") as f: f.write("\n".join(mbr))
with open("qe_rerank.txt", "w") as f: f.write("\n".join(qe_rerank))

print("Done. Transfer these 5 files to Colab.")
