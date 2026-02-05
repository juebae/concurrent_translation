"""Local WMT QE dataset loader"""
import datasets

class WMTQE(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(features=datasets.Features({
            "source": datasets.Value("string"),
            "target": datasets.Value("string"),
            "score": datasets.Value("float32"),
        }))
    
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": ("qe-dev-2022.en-de.src.tsv", "qe-dev-2022.en-de.mt.tsv", "qe-dev-2022.en-de.spans.tsv")}
            )
        ]
    
    def _generate_examples(self, files):
        src_file, mt_file, spans_file = files
        with open(src_file) as f_src, open(mt_file) as f_mt, open(spans_file) as f_spans:
            for id_, (src_line, mt_line, spans_line) in enumerate(zip(f_src, f_mt, f_spans)):
                # Simple score: 1.0 - (bad_spans_ratio)
                bad_spans = spans_line.strip().split() if spans_line.strip() else []
                score = max(0.0, 1.0 - len(bad_spans)/100.0)  # Normalized score
                yield id_, {"source": src_line.strip(), "target": mt_line.strip(), "score": score}
