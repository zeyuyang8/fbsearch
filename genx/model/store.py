from abc import ABC, abstractmethod

import numpy as np
from genx.model.llama import GenXTransformer


class Document:
    def __init__(self, text, metadata):
        self._text = text
        self._metadata = metadata

    def get_text(self):
        return self._text

    def get_metadata(self):
        return self._metadata


class IndexStoreTemplate(ABC):
    def __init__(self, initial_capacity=1000):
        # Capacity and initial capacity
        self._initial_capacity = initial_capacity
        self.capacity = initial_capacity

        # Data and global index of elements
        self.next_id = 0
        self.size = 0
        self._ids = np.zeros(self.capacity, dtype=np.int64)
        self._data_store = {}  # Dictionary to store actual data

    def _resize_if_needed(self, additional_items=16):
        if self.size + additional_items > self.capacity:
            new_capacity = max(self.capacity * 2, self.size + additional_items)

            # Resize ID array
            new_ids = np.zeros(new_capacity, dtype=np.int64)
            new_ids[: self.size] = self._ids[: self.size]
            self._ids = new_ids

            self.capacity = new_capacity

    def _clear_store(self):
        self.capacity = self._initial_capacity
        self.next_id = 0
        self.size = 0
        self._ids = np.zeros(self._initial_capacity, dtype=np.int64)
        self._data_store = {}

    def retrieve(self, doc_id):
        return self._data_store[doc_id]

    @abstractmethod
    def insert(self, text: list[Document]):
        pass

    @abstractmethod
    def query(self, query_text: Document) -> list[list[int]]:
        pass


class PrefixTreeNode:
    def __init__(self):
        self.children = {}
        self.doc_ids = set()


class Prompt:
    def __init__(self, before, after):
        self.before = before
        self.after = after

    def template(self, text):
        return self.before + text + self.after


class SequencePrefixTreeIndexStore(IndexStoreTemplate):
    def __init__(
        self,
        transformer,
        id_len,
        universe,
        doc_prompt_before="Generate identifying phrases that memorize the key concepts in this text. You are not supposed to make sense. Just generate ONLY the identifying phrases without any punctuations or numbers before or after. ",
        doc_prompt_after=" IGNORE ME. Phrases: ",
        query_prompt_before="From this query create identifying phrases that capture the key concepts and align with phrases found in relevant text. Do not aim for meaningful sentences. Only output the identifying phrases with no punctuation or numbers before or after. ",
        query_prompt_after=" IGNORE ME. Phrases: ",
        duplicate_prompt_before="Given the text first remove all the punctuations and stop words. Then shuffle the sentences. Generate some unique related phrases that does not have synonyms. ",
        duplicate_prompt_after=" IGNORE ME. Phrases: ",
        verbose=False,
        initial_capacity=1000,
        insertion_depth=3,
        mode="document_search",
    ):
        super().__init__(initial_capacity)

        assert mode in ["duplicate_detection", "document_search"]
        self.mode = mode
        if mode == "duplicate_detection":
            self.doc_prompt = Prompt(duplicate_prompt_before, duplicate_prompt_after)
            self.query_prompt = Prompt(duplicate_prompt_before, duplicate_prompt_after)
        elif mode == "document_search":
            self.doc_prompt = Prompt(doc_prompt_before, doc_prompt_after)
            self.query_prompt = Prompt(query_prompt_before, query_prompt_after)

        self.doc_prompt_before = doc_prompt_before
        self.doc_prompt_after = doc_prompt_after
        self.query_prompt_before = query_prompt_before
        self.query_prompt_after = query_prompt_after
        self.duplicate_prompt_before = duplicate_prompt_before
        self.duplicate_prompt_after = duplicate_prompt_after

        # Model for generating indices for inserted documens
        self.transformer: GenXTransformer = transformer
        self.id_len = id_len
        self.universe = set(universe)

        # Verbose
        self.verbose = verbose

        # Prefix tree
        self.root = PrefixTreeNode()
        self.insertion_depth = insertion_depth

    def set_verbose_for_all(self, verbose):
        self.verbose = verbose
        if hasattr(self.transformer, "verbose"):
            self.transformer.verbose = verbose

    def reset_id_len(self, id_len):
        self.id_len = id_len
        self.transformer.update_num_next_tokens(max_new_tokens=id_len)

    def set_mode(self, mode):
        assert mode in ["duplicate_detection", "document_search"]
        self.mode = mode
        if mode == "duplicate_detection":
            self.doc_prompt = Prompt(
                self.duplicate_prompt_before, self.duplicate_prompt_after
            )
            self.query_prompt = Prompt(
                self.duplicate_prompt_before, self.duplicate_prompt_after
            )
        elif mode == "document_search":
            self.doc_prompt = Prompt(self.doc_prompt_before, self.doc_prompt_after)
            self.query_prompt = Prompt(
                self.query_prompt_before, self.query_prompt_after
            )
        print("Remember to call `clear_store` to reset the database!")

    def clear_store(self):
        self.root = PrefixTreeNode()

        super()._clear_store()
        if self.verbose:
            print(f"Store cleared, current capacity: {self.capacity}")

    def _insert_document(self, texts: list[Document], prompt_template):
        if not isinstance(texts, list):
            texts = [texts]

        self._resize_if_needed(len(texts))

        doc_ids = []
        template_texts = []
        for text in texts:
            doc_id = self.next_id
            doc_ids.append(doc_id)
            # Update index in data store
            self.next_id += 1
            self.size += 1

            # Save text in data store
            self._ids[self.size - 1] = doc_id
            self._data_store[doc_id] = text

            template_text = prompt_template(text.get_text())
            template_texts.append(template_text)

        # Generate beams of sequences
        # [batch_size, num_return_sequences, sequence_length]
        lst_of_sequences = self.transformer.index_doc(template_texts)
        print(lst_of_sequences) if self.verbose else None
        self._insert_sequences_into_tree(lst_of_sequences, doc_ids)

    def _insert_sequences_into_tree(
        self, lst_of_sequences: list[list[list[int]]], doc_ids: list[int]
    ):
        for sequences, doc_id in zip(lst_of_sequences, doc_ids):
            for seq in sequences:
                print(f"Tokens: {seq}") if self.verbose else None
                if len(seq) != self.id_len or not all(x in self.universe for x in seq):
                    continue  # Skip invalid sequences

                self._traverse_and_insert(seq, doc_id)

    def _traverse_and_insert(self, seq, doc_id):
        node = self.root
        depth = 0

        for idx in seq:
            if idx not in node.children:
                node.children[idx] = PrefixTreeNode()
            node = node.children[idx]
            depth += 1
            if depth >= self.insertion_depth:
                node.doc_ids.add(doc_id)
                if self.verbose:
                    print(f"Inserted doc {doc_id} at depth {depth} of prefix tree")

    def insert(self, texts: list[Document]):
        print(f"Inserting '{texts}'") if self.verbose else None
        self._insert_document(texts, self.doc_prompt.template)

    def _query_with_prompt(self, query_texts: list[Document], prompt_template):
        if not query_texts:
            return []

        lst_of_result_ids = []
        template_texts = []
        for query_text in query_texts:
            template_text = prompt_template(query_text.get_text())
            print(template_text) if self.verbose else None
            template_texts.append(template_text)

        # [batch_size, num_return_sequences, sequence_length]
        lst_of_sequences = self.transformer.index_query(template_texts)
        for sequences in lst_of_sequences:
            result_ids = []
            for seq in sequences:
                print(f"Tokens: {seq}") if self.verbose else None
                if len(seq) != self.id_len or not all(x in self.universe for x in seq):
                    continue

                result = self._traverse_tree_for_query(seq)
                if result:
                    result["index_ids"] = seq
                    result["index_txt"] = self.transformer.doc_tokenizer.batch_decode(
                        seq, skip_special_tokens=False
                    )
                    result_ids.append(result)
            print("Found results: ", result_ids) if self.verbose else None
            lst_of_result_ids.append(result_ids)

        return lst_of_result_ids

    def _traverse_tree_for_query(self, seq):
        node: PrefixTreeNode = self.root
        found = True
        depth = 0

        for idx in seq:
            if (idx not in node.children) and (depth < self.insertion_depth):
                found = False
                break
            if (idx not in node.children) and (depth >= self.insertion_depth):
                break
            node = node.children[idx]
            depth += 1

        if found:
            return {"depth": depth, "doc_ids": node.doc_ids}
        return None

    def query(self, query_texts: list[Document]):
        print(f"Querying for '{query_texts}'") if self.verbose else None
        return self._query_with_prompt(query_texts, self.query_prompt.template)
