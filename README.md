<div align="center">

# UDA-Benchmark

A Benchmark Suite for Retrieval Augmented Generation in Real-world Document Analysis
</div>


## Introduction
UDA (Unstructured Document Analysis) is a benchmark suite for Retrieval Augmented Generation (RAG) in real-world document analysis. Each entry in the UDA dataset is organized as a *document-question-answer* triplet, where a question is raised from the document, accompanied by a corresponding ground-truth answer. The documents are retained in their original file formats without parsing or segmentation; they consist of both textual and tabular data, reflecting the complex nature of real-world analytical scenarios.


## Quick Start

Begin by setting up the necessary libraries and source codes:
```shell
git clone git@github.com:qinchuanhui/UDA-Benchmark.git 
cd UDA-Benchmark
pip install -r requirements.txt
```

For a quick introduction to the functionalities of our benchmark suite, refer to the [basic_demo.ipynb](basic_demo.ipynb) notebook. It outlines a standard workflow for document analysis using our UDA-QA dataset. 

The demonstration encompasses several essential steps:
 * Prepare the question-answer-document triplet
 * Extract and segment the document content
 * Build indexes and retrieve data segments
 * Generate answering reponse with LLMs 

## Dataset: UDA-QA 

### Description
 Each entry within the UDA dataset is organized as a document-question-answer pair. A typical data point may look like:
```python
{ 'doc_name': 'ADI_2009', # a financial report
  'q_uid': 'ADI/2009/page_59.pdf-2',  # unique question id
  'question': 'What is the expected growth rate in amortization expense in 2010?',
  'answer_1': '-27.0%',
  'answer_2': '	-0.26689'}
```

The UDA dataset comprises six subsets spanning finance, academia, and knowledge bases, encompassing 2965 documents and 29590 expert-annotated Q&A pairs (more details in [HuggingFace](https://huggingface.co/datasets/qinchuanhui/UDA-QA)). The following table shows an overview of sub-datasets in UDA and their statistics. 

| Sub Dataset <br />(Source Domain) | Doc Format | Doc Num | Q&A Num | Avg #Words | Avg #Pages | Total Size | Q&A Types                        |
| --------------------------------- | ---------- | ------- | ------- | ---------- | ---------- | ---------- | -------------------------------- |
| FinHybrid (Finance)               | PDF        | 788     | 8190    | 76.6k      | 147.8      | 2.61 GB    | arithmetic                       |
| TatHybrid (Finance)               | PDF        | 170     | 14703   | 77.5k      | 148.5      | 0.58 GB    | extractive, counting, arithmetic |
| PaperTab (Academia)               | PDF        | 307     | 393     | 6.1k       | 11.0       | 0.22 GB    | extractive, yes/no, free-form    |
| PaperText (Academia)              | PDF        | 1087    | 2804    | 5.9k       | 10.6       | 0.87 GB    | extractive, yes/no, free-form    |
| FetaTab (Wikipedia)               | PDF & HTML | 878     | 1023    | 6.0k       | 14.9       | 0.92 GB    | free-form                        |
| NqText (Wikipedia)                | PDF & HTML | 645     | 2477    | 6.1k       | 14.9       | 0.68 GB    | extractive                       |

### Dataset Usage

For the Q&A labels, they are accessible either through the csv files in the [dataset/qa](dataset/qa/) directory or by loading the dataset from the HuggingFace repository `qinchuanhui/UDA-QA`. The basic usage and format conversion can be found in [basic_demo.ipynb](basic_demo.ipynb).

To access the source document files, you can download the complete set through the [HuffingFace Repo](https://huggingface.co/datasets/qinchuanhui/UDA-QA/tree/main/src_doc_files). After downloading, extract the content into [dataset/src_doc_files](dataset/src_doc_files).  For illustrative purposes, some examples of source documents can be found in  [dataset/src_doc_file_example](dataset/src_doc_files_example/).

Additionally, we also include some extended information related to question-answering tasks, which encompass  reasoning explanations, human-validated factual evidence, and structured contexts. These resources are pivotal for an in-depth analysis of the modular benchmark. You can obtain the full set of them from the [HuffingFace Repo](https://huggingface.co/datasets/qinchuanhui/UDA-QA/tree/main/extended_qa_info), and place them in [dataset/extended_qa_info](dataset/extended_qa_info/).  A sampled subset for evaluation purposes is also available in  [dataset/extended_qa_info_bench](dataset/extended_qa_info_bench/).







## Benchmark and Experiments

Our UDA benchmark focuses on several pivotal items:

* The effectiveness of various table-parsing approaches
* The performance of different indexing and retrieval strategies
* The influence of precise retrieval on the quality of LLM interpretation
* The effectiveness of long-context LLMs compared to typical RAGs
* Comparison of different LLM-based Q&A strategies
* End-to-end comparisons of various LLMs across diverse applications 

### Evaluation Metrics
To evaluate the quality of LLM-generated answers, we apply widely accepted span-level F1-score in PaperTab, PaperText, FetaTab, and NqText datasets, where ground-truth answers are in natural language and the source datasets also utilize this metric. We treat the prediction and ground truth as bags of words and calculate the F1-score to measure their overlap (see [basic_eval](uda/eval/utils/basic_utils.py)). 

In financial analysis, the assessment becomes more intricate due to numerical values. For the TatHybrid dataset, we adopt the numeracy-focused F1-score, which considers the scale and the plus-minus of numerical values (see [financial_eval](uda/eval/utils/finance_utils.py)). In the FinHybrid dataset, where answers are always numerical or binary, we rely on the Exact-Match metric but allow for a numerical tolerance of $1\%$, accounting for rounding discrepancies. 

For more details of our benchmark and experiments, please refer to the directory [exp](exp).



## Licenses

[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

Our UDA dataset is distributed under the Creative Commons Attribution-ShareAlike 4.0 International ([CC-BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)) License.

[![CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
