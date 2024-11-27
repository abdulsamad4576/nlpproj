import uuid
import os
import re
from typing import List, Tuple, Union, Dict, Any
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from utils import compute_time, connect_to_db, create_collection_if_not_exists, add_vectors, generate_vector,extract_step_ids
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
CHUNK_SIZE = 50  # Chunk size in tokens
PAGE_SIZE = 200   # Page size in words

# High-level class for program-related operations
class VectorDBManager:
    @compute_time
    def __init__(self, connection_string: str, api_key: str, index_dir: str = "indexdir"):
        self.db = connect_to_db(connection_string, api_key)
        self.index_dir = index_dir
        self._setup_whoosh_index()

    @compute_time
    def clear_collection(self, collection_name: str):
        """
        Clears all vectors and metadata in the specified Qdrant collection.
        """
        try:
            response: CollectionOperationResponse = self.db.delete_collection(collection_name=collection_name)
            print(f"Collection {collection_name} cleared.")
            return response
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    # Setup Whoosh index
    @compute_time
    def _setup_whoosh_index(self):
        schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
            self.index = create_in(self.index_dir, schema)
            print("Whoosh index created.")
        else:
            self.index = open_dir(self.index_dir)
            print("Whoosh index opened.")

    # Add document to Whoosh index
    @compute_time
    def _add_to_whoosh_index(self, doc_id: str, content: str):
        writer = self.index.writer()
        writer.add_document(id=doc_id, content=content)
        writer.commit()

    @compute_time
    def create_collection(self, collection_name: str, vector_size: int):
        create_collection_if_not_exists(self.db, collection_name, vector_size)

    @compute_time
    def delete_program(self,collection_name: str, program_id: str):
        print(f"Deleting the entire program: {program_id}")
        offset = None
        # Step 1: Fetch all steps associated with the program
        try:
            while True:
                results, next_offset = self.db.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="program_id", match=MatchValue(value=program_id))]
                    ),
                    with_payload=True,
                    offset=offset
                )
                if next_offset is None or len(results) == 0:
                    break

                offset = next_offset  # Update offset to fetch the next batch

        except Exception as e:
            print(f"Error fetching program steps: {e}")
            return False

        # Step 2: Iterate through the steps and delete them
        step_ids = set()
        for result in results:
            step_id = result.payload.get("step_id")
            if step_id and step_id not in step_ids:
                self.delete_program_step(collection_name, program_id, step_id)
                step_ids.add(step_id)

        print(f"Program {program_id} and all its steps have been deleted.")
        return True

    @compute_time
    def get_step_id_for_program(self,collection_name: str, program_id: str, step_name: str) -> Union[int, None]:
        program_id=int(program_id)

        try:
            offset = None  # Initialize offset
            while True:
                results, next_offset = self.db.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="program_id", match=MatchValue(value=program_id)),
                            FieldCondition(key="step_name", match=MatchValue(value=step_name)),
                            FieldCondition(key="level", match=MatchValue(value=2))  # Level 2 represents the step level
                        ]
                    ),
                    with_payload=True,
                    offset=offset  # Continue from the last position
                )
                
                # If no more results or next_offset is None, break the loop
                if next_offset is None or len(results) == 0:
                    break

                offset = next_offset  # Update offset to fetch the next batch

        except Exception as e:
            print(f"Error during scroll: {e}")


            # Extract step_id from the results
            if results:
                step_id = results[0].payload.get('step_id', None)
                if step_id:
                    print(f"Found step_id: {step_id} for step_name: {step_name}")
                    return int(step_id)
                else:
                    print(f"No step_id found for step_name: {step_name}")
                    return None
            else:
                print(f"No results found for program_id: {program_id}, step_name: {step_name}")
                return None

        except Exception as e:
            print(f"Error fetching step_id: {e}")
            return None
    # Utility function to get sibling chunks for context
    @compute_time
    def _get_sibling_chunks(self, chunks: List[str], current_chunk_num: int) -> Dict[str, str]:
        siblings = {}
        if current_chunk_num > 1:
            siblings["previous_chunk"] = chunks[current_chunk_num - 2]  # Previous chunk
        if current_chunk_num < len(chunks):
            siblings["next_chunk"] = chunks[current_chunk_num]  # Next chunk
        return siblings
    @compute_time
    def edit_step(self, collection_name: str, program_id: str, step_id: int, new_step_data: Dict[str, Any]):
        
        program_id = int(program_id)

        # Step 1: Retrieve the existing step name using the step_id
        existing_step = self.db.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="program_id", match=MatchValue(value=program_id)),
                    FieldCondition(key="step_id", match=MatchValue(value=step_id))
                ]
            ),
            with_payload=True,
            limit=1
        )
        
        if not existing_step[0]:
            print(f"Step with ID {step_id} not found.")
            return False
        
        
        # Step 2: Delete the old step using step_id
        self.delete_program_step(collection_name, program_id, step_id)
        
        # Step 3: Add the new step with the same step name using the add_program_step function
        self.add_program_step(collection_name, program_id,  new_step_data,step_id)

        print(f"Step with ID '{step_id}' updated successfully.")
        return True


    @compute_time
    def delete_program_step(self,collection_name: str, program_id: int, step_id: int): 
        print(f"Deleting Step ID: {step_id} from program: {program_id}")
        
        # Use FilterSelector to delete points based on step_id
        points_selector = FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(key="program_id", match=MatchValue(value=program_id)),
                    FieldCondition(key="step_id", match=MatchValue(value=step_id))
                ]
            )
        )
        print(f"Deleting points with step_id: {step_id}")
        print(f"Points Selector: {points_selector}")
        # Check if points_selector is None
        if points_selector is None:
            print("No points to delete")
            return

        # Remove all vectors related to the step from Qdrant
        self.db.delete(
            collection_name=collection_name,
            points_selector=points_selector
        )

        # Use Whoosh for keyword-based content search and deletion
        # Find all chunks for the step in the Whoosh index and delete them
        with self.index.searcher() as searcher:
            query = QueryParser("content", self.index.schema).parse(str(step_id))
            whoosh_results = searcher.search(query)

            for hit in whoosh_results:
                chunk_id = hit['id']
                # Delete chunk from Whoosh
                self._delete_from_whoosh_index(chunk_id)
                self._delete_from_whoosh_index(f"{chunk_id}_context")
        
        print(f"Step ID '{step_id}' deleted successfully from Qdrant and Whoosh.")

    @compute_time
    def update_steps(self,collection_name: str, program_id: str, added_steps: List[Dict[str, Any]], removed_step_ids: List[int]):

        # Step 1: Remove old steps
        for step_id in removed_step_ids:
            print(f"Removing step with ID: {step_id}")
            self.delete_program_step(collection_name, program_id, step_id)

        # Step 2: Add new steps
        for step_data in added_steps:
            print(f"Adding new step with id: {step_data['step_id']}")
            self.add_program_step(collection_name, program_id, step_data)

        print("Steps update complete.")
        return True
    @compute_time
    def add_program_step(self,collection_name: str, program_id: str, step_data: Dict[str, Any], step_id: int = None):
        step_id = step_id if step_id is not None else int(uuid.uuid4().hex[:8], 16)
        print("step data content",step_data["content"])
        content = step_data['content']
        pages = self._split_into_pages(content)
        program_id = int(program_id)

        for page_num, page_content in enumerate(pages, start=1):
            page_id = int(uuid.uuid4().hex[:8], 16)
            page_vector = generate_vector(page_content)
            page_metadata = {
                "level": 2,
                "program_id": program_id,
                "step_id": step_id,  
                "page_num": page_num,
            }
            page_point = PointStruct(id=page_id, vector=page_vector, payload=page_metadata)
            add_vectors(self.db, collection_name, [page_point])

            paragraphs = self._split_into_paragraphs(page_content)

            for paragraph_num, paragraph_content in enumerate(paragraphs, start=1):
                paragraph_id = int(uuid.uuid4().hex[:8], 16)
                paragraph_vector = generate_vector(paragraph_content)
                paragraph_metadata = {
                    "level": 3,
                    "program_id": program_id,
                    "step_id": step_id,  
                    "page_num": page_num,
                    "paragraph_num": paragraph_num,
                }
                paragraph_point = PointStruct(id=paragraph_id, vector=paragraph_vector, payload=paragraph_metadata)
                add_vectors(self.db, collection_name, [paragraph_point])

                chunks = self._split_into_chunks(paragraph_content)

                for chunk_num, chunk_content in enumerate(chunks, start=1):
                    chunk_id = int(uuid.uuid4().hex[:8], 16)
                    chunk_vector = generate_vector(chunk_content)
                    siblings = self._get_sibling_chunks(chunks, chunk_num)
                    context = {
                        "current_chunk": chunk_content,
                        "previous_chunk": siblings.get("previous_chunk", ""),
                        "next_chunk": siblings.get("next_chunk", "")
                    }
                    chunk_metadata = {
                        "level": 4,
                        "program_id": program_id,
                        "step_id": step_id,  
                        "page_num": page_num,
                        "paragraph_num": paragraph_num,
                        "chunk_num": chunk_num,
                        "context": context
                    }
                    chunk_point = PointStruct(id=chunk_id, vector=chunk_vector, payload=chunk_metadata)
                    add_vectors(self.db, collection_name, [chunk_point])

                    self._add_to_whoosh_index(str(chunk_id), chunk_content)
                    self._add_to_whoosh_index(str(chunk_id) + "_context", context["previous_chunk"] + " " + context["next_chunk"])

        print(f"Step Id '{step_id}' added successfully with ID: {step_id}.")
    @compute_time
    def add_program(self, collection_name: str, program_id: str, program_data: Dict[str, Any], chunk_size: int, page_size: int) -> bool:
        """
        Adds a program and its steps to the collection.
        :param chunk_size: Size of the chunks in tokens.
        :param page_size: Size of the pages in words.
        """
        global CHUNK_SIZE, PAGE_SIZE
        CHUNK_SIZE = chunk_size
        PAGE_SIZE = page_size

        # Add or update the steps that have been added or modified
        step_ids = extract_step_ids(program_data)
        for i, (step_name, step_data) in enumerate(program_data["steps"].items()):
            self.add_program_step(collection_name, program_id, step_data, step_ids[i])

        return True




    # @compute_time
    # def hierarchical_search(self, query: str, collection_name: str, program_id: str, limit: int = 5, step_id: int = None) -> List[Dict[str, Any]]:
    #     #refined_query = self._refine_query_with_llm(query)
    #     print(f"Original Query: '{query}'")
    #     #print(f"Refined Query: '{refined_query}'")
    #     refined_query=query
    #     program_id = int(program_id)
    #     query_vector = generate_vector(refined_query)

    #     try:
    #         # Add a filter for step_id if provided
    #         step_filter = []
    #         if step_id:
    #             step_filter.append(FieldCondition(key="step_id", match=MatchValue(value=step_id)))
    #             print(f"Searching within step_id: {step_id}")

    #         # Step 1: Search at Page Level
    #         print(f"Searching for pages with program_id: {program_id}")
    #         page_search = self.db.search(
    #             collection_name=collection_name,
    #             query_vector=query_vector,
    #             limit=limit * 5,
    #             with_payload=True,
    #             query_filter=Filter(
    #                 must=[
    #                     FieldCondition(key="program_id", match=MatchValue(value=program_id)),
    #                     FieldCondition(key="level", match=MatchValue(value=2)),
    #                     *step_filter  # Add the step_id filter if provided
    #                 ]
    #             ),
    #         )
    #         page_step_ids = [result.payload['step_id'] for result in page_search]
    #         top_pages = [result.payload['page_num'] for result in page_search]
    #         for i in range(len(top_pages)):
    #             print(f"Page {top_pages[i]} has step_id: {page_step_ids[i]}")
    #         if not top_pages:
    #             print("No pages found at the page level.")
    #             return []

    #         # Step 2: Parallel Search at Paragraph Level using ThreadPoolExecutor
    #         print("Searching paragraphs in parallel...")
    #         top_paragraphs = []
    #         with ThreadPoolExecutor() as executor:
    #             paragraph_futures = {
    #                 executor.submit(
    #                     self.db.search,
    #                     collection_name=collection_name,
    #                     query_vector=query_vector,
    #                     limit=limit * 3,
    #                     with_payload=True,
    #                     query_filter=Filter(
    #                         must=[
    #                             FieldCondition(key="program_id", match=MatchValue(value=program_id)),
    #                             FieldCondition(key="page_num", match=MatchValue(value=page_num)),
    #                             FieldCondition(key="level", match=MatchValue(value=3)),
    #                             *step_filter  # Add the step_id filter if provided
    #                         ]
    #                     )
    #                 ): page_num for page_num in top_pages
    #             }
    #             # Collect results from paragraph futures as they complete
    #             for future in as_completed(paragraph_futures):
    #                 top_paragraphs.extend(future.result())
    #         # Sort paragraphs based on page and paragraph numbers
    #         top_paragraphs = sorted(top_paragraphs, key=lambda x: (x.payload['page_num'], x.payload['paragraph_num']))
    #         if not top_paragraphs:
    #             print("No paragraphs found within the pages.")
    #             return []

    #         # Step 3: Parallel Search at Chunk Level using ThreadPoolExecutor
    #         print("Searching chunks in parallel...")
    #         top_chunks = []
    #         with ThreadPoolExecutor() as executor:
    #             chunk_futures = {
    #                 executor.submit(
    #                     self.db.search,
    #                     collection_name=collection_name,
    #                     query_vector=query_vector,
    #                     limit=limit * 2,
    #                     with_payload=True,
    #                     query_filter=Filter(
    #                         must=[
    #                             FieldCondition(key="program_id", match=MatchValue(value=program_id)),
    #                             FieldCondition(key="page_num", match=MatchValue(value=paragraph_result.payload['page_num'])),
    #                             FieldCondition(key="paragraph_num", match=MatchValue(value=paragraph_result.payload['paragraph_num'])),
    #                             FieldCondition(key="level", match=MatchValue(value=4)),
    #                             *step_filter  # Add the step_id filter if provided
    #                         ]
    #                     )
    #                 ): paragraph_result for paragraph_result in top_paragraphs
    #             }
    #             # Collect results from chunk futures as they complete
    #             for future in as_completed(chunk_futures):
    #                 top_chunks.extend(future.result())
    #         # Sort chunks based on page, paragraph, and chunk numbers
    #         top_chunks = sorted(top_chunks, key=lambda x: (x.payload['page_num'], x.payload['paragraph_num'], x.payload['chunk_num']))
    #         if not top_chunks:
    #             print("No chunks found within the paragraphs.")
    #             return []

    #         # Step 4: Keyword-based search using Whoosh
    #         keyword_results = self._keyword_search(refined_query, top_n=limit * 10)

    #         # Map context IDs back to their chunk IDs
    #         keyword_chunk_ids = set()
    #         for hit in keyword_results:
    #             hit_id = hit['id']
    #             chunk_id = hit_id[:-8] if hit_id.endswith("_context") else hit_id
    #             keyword_chunk_ids.add(chunk_id)

    #         # Step 5: Combine chunk-level results and boost scores for keyword matches
    #         combined_results = {}
    #         for result in top_chunks:
    #             chunk_id = result.id
    #             vector_score = result.score
    #             combined_score = vector_score + 1.0 if chunk_id in keyword_chunk_ids else vector_score
    #             combined_results[chunk_id] = {
    #                 "combined_score": combined_score,
    #                 "vector_score": vector_score,
    #                 "payload": result.payload,
    #             }

    #         # Extract minimum and maximum combined scores
    #         all_scores = [result['combined_score'] for result in combined_results.values()]
    #         min_score = min(all_scores)
    #         max_score = max(all_scores)

    #         # Normalize the scores between 0 and 1
    #         def normalize_score(score):
    #             if max_score - min_score == 0:
    #                 return 1.0  # Avoid division by zero in case all scores are the same
    #             return (score - min_score) / (max_score - min_score)

    #         # Manually sort the results based on the normalized score
    #         sorted_combined_results = sorted(combined_results.items(), key=lambda x: normalize_score(x[1]['combined_score']), reverse=True)

    #         # Prepare final results based on the limit
    #         final_results = []
    #         for chunk_id, result_info in sorted_combined_results[:limit]:
    #             chunk_metadata = result_info["payload"]
    #             chunk_content = chunk_metadata.get("current_chunk", "")
    #             context = chunk_metadata.get("context", {})
    #             final_results.append({
    #                 "id": chunk_id,
    #                 "score": normalize_score(result_info['combined_score']),
    #                 "metadata": chunk_metadata,
    #                 "content": chunk_content,
    #                 "context": context
    #             })

    #         return final_results

    #     except Exception as e:
    #         print(f"Error during hierarchical search: {e}")
    #         return []
    @compute_time
    def hierarchical_search(self, query: str, collection_name: str, program_id: str, limit: int = 5, step_id: int = None) -> List[Dict[str, Any]]:
        print(f"Original Query: '{query}'")
        refined_query = query  # Keeping the query as is for now
        program_id = int(program_id)
        query_vector = generate_vector(refined_query)

        try:
            # Add a filter for step_id if provided
            step_filter = []
            if step_id:
                step_filter.append(FieldCondition(key="step_id", match=MatchValue(value=step_id)))
                print(f"Searching within step_id: {step_id}")

            # Step 1: Search at Page Level
            print(f"Searching for pages with program_id: {program_id}")
            page_search = self.db.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit * 5,
                with_payload=True,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="program_id", match=MatchValue(value=program_id)),
                        FieldCondition(key="level", match=MatchValue(value=2)),
                        *step_filter  # Add the step_id filter if provided
                    ]
                ),
            )
            if not page_search:
                print("No pages found at the page level.")
                return []

            # Collect results from higher levels
            top_chunks = []  # Final collection of chunks to consider
            for page_result in page_search:
                page_num = page_result.payload["page_num"]
                paragraphs = self.db.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit * 3,
                    with_payload=True,
                    query_filter=Filter(
                        must=[
                            FieldCondition(key="program_id", match=MatchValue(value=program_id)),
                            FieldCondition(key="page_num", match=MatchValue(value=page_num)),
                            FieldCondition(key="level", match=MatchValue(value=3)),
                            *step_filter
                        ]
                    ),
                )
                for paragraph_result in paragraphs:
                    paragraph_num = paragraph_result.payload["paragraph_num"]
                    chunks = self.db.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        limit=limit * 2,
                        with_payload=True,
                        query_filter=Filter(
                            must=[
                                FieldCondition(key="program_id", match=MatchValue(value=program_id)),
                                FieldCondition(key="page_num", match=MatchValue(value=page_num)),
                                FieldCondition(key="paragraph_num", match=MatchValue(value=paragraph_num)),
                                FieldCondition(key="level", match=MatchValue(value=4)),
                                *step_filter
                            ]
                        ),
                    )
                    top_chunks.extend(chunks)

            # Normalize scores
            combined_results = {}
            for result in top_chunks:
                chunk_id = result.id
                vector_score = result.score
                combined_results[chunk_id] = {
                    "vector_score": vector_score,
                    "payload": result.payload
                }

            # Extract vector scores
            vector_scores = [result["vector_score"] for result in combined_results.values()]
            min_vector_score = min(vector_scores) if vector_scores else 0.0
            max_vector_score = max(vector_scores) if vector_scores else 1.0

            # Normalize scores and filter results
            final_results = []
            for chunk_id, result_info in combined_results.items():
                normalized_score = (
                    (result_info["vector_score"] - min_vector_score) / (max_vector_score - min_vector_score)
                    if max_vector_score > min_vector_score else 0.0
                )
                if normalized_score >= 0.5:  # Filter by a relevance threshold
                    chunk_metadata = result_info["payload"]
                    chunk_content = chunk_metadata.get("current_chunk", "")
                    context = chunk_metadata.get("context", {})
                    final_results.append({
                        "id": chunk_id,
                        "score": normalized_score,
                        "metadata": chunk_metadata,
                        "content": chunk_content,
                        "context": context,
                    })

            # Sort final results by score
            final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:limit]

            return final_results

        except Exception as e:
            print(f"Error during hierarchical search: {e}")
            return []


    # # Custom reranking using GPT
    # @compute_time
    # def _rerank_with_gpt(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """
    #     Rerank the documents using GPT, return the documents with assigned scores.
    #     """
    #     # Use the provided system prompt
    #     system_prompt = """
    #     You are a highly capable assistant. Your task is to rerank the following document chunks based on their relevance to the user's query. 
    #     The goal is to return a ranked list where the best matches appear first.
    #     Assign a score between 0 and 1 to each document, with 1 being the most relevant.
    #     Ensure that each document is evaluated uniquely, and no document is mentioned more than once.
        
    #     Query: {query}
        
    #     Documents to rerank:
    #     {documents}
        
    #     Please provide a score for each document in the following format:
    #     Document 1: Score 0.8 - [Brief explanation]
    #     Document 2: Score 0.6 - [Brief explanation]
    #     Document 3: Score 0.4 - [Brief explanation]
    #     """

    #     # Format the documents for inclusion in the prompt
    #     documents_text = ""
    #     for idx, doc in enumerate(documents, start=1):
    #         documents_text += f"Document {idx}:\n{doc['text']}\n\n"

    #     # Now, format the system prompt with the query and documents
    #     system_prompt_formatted = system_prompt.format(query=query, documents=documents_text)

    #     # Create the messages for ChatCompletion
    #     messages = [
    #         {"role": "system", "content": system_prompt_formatted}
    #     ]

    #     # Call the ChatCompletion API
    #     try:
    #         response = client.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             messages=messages,
    #             max_tokens=500,
    #             temperature=0.3,  # Lower temperature for more deterministic responses
    #         )
    #         assistant_message = response.choices[0].message.content.strip()

    #         # Parse the assistant_message to extract the scores
    #         reranked_documents = []
    #         lines = assistant_message.split('\n')
    #         for line in lines:
    #             # Use regex to extract document number and score
    #             match = re.match(r'Document\s+(\d+):\s+Score\s+([0-9.]+)\s*-\s*(.*)', line)
    #             if match:
    #                 doc_num = int(match.group(1)) - 1  # Adjust to zero-based index
    #                 score = float(match.group(2))
    #                 # Get the corresponding document
    #                 if 0 <= doc_num < len(documents):
    #                     doc = documents[doc_num]
    #                     reranked_documents.append({
    #                         'id': doc['id'],
    #                         'score': score,
    #                         'text': doc['text']
    #                     })
    #             else:
    #                 # If the line doesn't match, skip it
    #                 continue

    #         # Sort the documents by score in descending order
    #         reranked_documents.sort(key=lambda x: x['score'], reverse=True)
    #         return reranked_documents
    #     except Exception as e:
    #         print(f"Error during reranking with GPT: {e}")
    #         # In case of error, return the documents without reranking
    #         return documents


    # Query Transformation Using LLM
    @compute_time
    def _refine_query_with_llm(self, query: str) -> str:
        system_prompt = """
        You are an intelligent assistant that helps improve search queries for retrieving information from hierarchical documents. 
        The documents are split into steps, pages, paragraphs, and chunks. Each chunk contains specific information that could match a user query. 
        Your goal is to refine the user's query to make it more specific, detailed, and suitable for document retrieval, while preserving the original intent. 
        Consider expanding keywords, resolving ambiguities, and adding relevant context to improve the match with the stored content.

        Examples:
        1. Original: "intro to system" -> Refined: "introduction to the system, covering key components and functionality"
        2. Original: "details technical" -> Refined: "technical details of the hierarchical  sembeddingystem, including implementation and optimizations"

        Please refine the following search query:
        """
        user_input = f"Original Query: '{query}'"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=32,
                temperature=0.5,
                n=1,
                stop=None,
            )
            refined_query = response.choices[0].message.content.strip()
            return refined_query
        except Exception as e:
            print(f"Error during query refinement: {e}")
            return query  # Return the original query if an error occurs

    # Keyword-based search using Whoosh
    @compute_time
    def _keyword_search(self, query_str: str, top_n: int = 50) -> List[Dict[str, Any]]:
        parser = QueryParser("content", schema=self.index.schema)
        query = parser.parse(query_str)
        results = []
        with self.index.searcher() as searcher:
            whoosh_results = searcher.search(query, limit=top_n)
            for hit in whoosh_results:
                results.append({"id": hit["id"], "score": hit.score})
        return results

    @compute_time
    def _split_into_pages(self, content: str) -> List[str]:
        words = content.split()
        pages = [' '.join(words[i:i + PAGE_SIZE]) for i in range(0, len(words), PAGE_SIZE)]
        return pages

    # Utility function to split content into paragraphs
    @compute_time
    def _split_into_paragraphs(self, content: str) -> List[str]:
        # We ensure that paragraphs are separated by newlines or any common delimiter
        return [p.strip() for p in content.split('\n') if p.strip()]

    @compute_time
    def _split_into_chunks(self,content: str) -> List[str]:
        """
        Splits the given content into chunks of text based on defined separators and chunk size.
        """
        # Define primary separators to split the content
        separators = r"[!?\n;]"  # Regular expression for separators: newline, exclamation, question, and semicolon

        # Step 1: Split the content into initial chunks based on separators
        initial_chunks = re.split(separators, content)

        # Step 2: Further refine by splitting sentences on ". " where appropriate
        refined_chunks = []
        for chunk in initial_chunks:
            # Split by ". " only if it’s followed by an uppercase letter or digit
            sub_chunks = re.split(r'(?<!\d)\. (?=[A-Z0-9])', chunk.strip())
            refined_chunks.extend(sub_chunks)

        # Step 3: Combine small chunks to ensure they meet the minimum CHUNK_SIZE
        final_chunks = []
        temp_chunk = ""
        for sentence in refined_chunks:
            # If adding the sentence doesn't exceed CHUNK_SIZE, append to temp_chunk
            if len(temp_chunk.split()) + len(sentence.split()) <= CHUNK_SIZE or temp_chunk == "":
                temp_chunk += sentence.strip() + " "
            else:
                # Otherwise, commit the current temp_chunk and start a new one
                final_chunks.append(temp_chunk.strip())
                temp_chunk = sentence.strip() + " "

        # Add any remaining text as the final chunk
        if temp_chunk.strip():
            final_chunks.append(temp_chunk.strip())

        return final_chunks

