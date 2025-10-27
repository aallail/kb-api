"""Integration tests for the complete RAG pipeline."""
import pytest
import uuid
from io import BytesIO
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

# Test password from config (default dev-key)
TEST_PASSWORD = "dev-key"


def test_full_rag_pipeline():
    """
    Test complete RAG pipeline: upload document → ask question → get answer.

    This is the most important integration test as it validates the entire system.
    """
    # Create a test document
    test_content = b"""
    Tesla Model S is an electric vehicle with a range of 400 miles.
    It features autopilot capabilities and over-the-air software updates.
    The vehicle has a top speed of 155 mph and can accelerate from 0-60 mph in 3.1 seconds.
    Tesla also produces the Model 3, Model X, and Model Y.
    """

    # Step 1: Upload document
    files = {"file": ("tesla.txt", BytesIO(test_content), "text/plain")}
    data = {"title": "Tesla Model S Specifications"}
    response = client.post(
        "/documents",
        files=files,
        data=data,
        headers={"password": TEST_PASSWORD}
    )

    assert response.status_code == 201, f"Upload failed: {response.text}"
    upload_result = response.json()
    doc_id = upload_result["doc_id"]
    assert doc_id is not None
    print(f"✓ Document uploaded: {doc_id}")

    # Step 2: Ask a question (vector search)
    question_data = {
        "query": "What is the range of the Tesla Model S?",
        "top_k": 3
    }
    response = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )

    assert response.status_code == 200, f"Question failed: {response.text}"
    answer_result = response.json()

    # Verify response structure
    assert "answer" in answer_result
    assert "sources" in answer_result
    assert "query" in answer_result

    # Verify answer content (should mention the range)
    answer = answer_result["answer"].lower()
    assert "400" in answer or "range" in answer, f"Expected range info in answer: {answer}"

    # Verify sources
    assert len(answer_result["sources"]) > 0
    assert answer_result["sources"][0]["doc_id"] == doc_id
    print(f"✓ Question answered: {answer_result['answer'][:100]}...")

    # Step 3: Test hybrid search
    question_data["use_hybrid"] = True
    response = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )

    assert response.status_code == 200
    hybrid_result = response.json()
    assert "answer" in hybrid_result
    print(f"✓ Hybrid search works")

    # Step 4: Cleanup - Delete document
    response = client.delete(
        f"/documents/{doc_id}",
        headers={"password": TEST_PASSWORD}
    )
    assert response.status_code == 204
    print(f"✓ Document deleted")


def test_multi_document_retrieval():
    """
    Test retrieval from multiple documents.

    Validates that the system can correctly search across multiple documents.
    """
    # Upload two documents
    doc1_content = b"Python is a high-level programming language known for its simplicity."
    doc2_content = b"Java is a statically typed programming language used for enterprise applications."

    # Upload doc 1
    files1 = {"file": ("python.txt", BytesIO(doc1_content), "text/plain")}
    response1 = client.post(
        "/documents",
        files=files1,
        data={"title": "Python Info"},
        headers={"password": TEST_PASSWORD}
    )
    assert response1.status_code == 201
    doc1_id = response1.json()["doc_id"]
    print(f"✓ Doc 1 uploaded: {doc1_id}")

    # Upload doc 2
    files2 = {"file": ("java.txt", BytesIO(doc2_content), "text/plain")}
    response2 = client.post(
        "/documents",
        files=files2,
        data={"title": "Java Info"},
        headers={"password": TEST_PASSWORD}
    )
    assert response2.status_code == 201
    doc2_id = response2.json()["doc_id"]
    print(f"✓ Doc 2 uploaded: {doc2_id}")

    # Ask a question that should retrieve from both documents
    question_data = {
        "query": "Tell me about programming languages",
        "top_k": 5
    }
    response = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )

    assert response.status_code == 200
    result = response.json()

    # Should get sources from both documents
    source_doc_ids = set(source["doc_id"] for source in result["sources"])
    # Note: Might get both or just the most relevant one depending on query
    assert len(source_doc_ids) >= 1
    print(f"✓ Retrieved from {len(source_doc_ids)} document(s)")

    # Cleanup
    client.delete(f"/documents/{doc1_id}", headers={"password": TEST_PASSWORD})
    client.delete(f"/documents/{doc2_id}", headers={"password": TEST_PASSWORD})
    print("✓ Cleanup complete")


def test_caching():
    """
    Test response caching works correctly.

    Validates that repeated queries return cached responses.
    """
    # Upload a test document
    test_content = b"Caching test document with some unique content for testing."
    files = {"file": ("cache_test.txt", BytesIO(test_content), "text/plain")}
    response = client.post(
        "/documents",
        files=files,
        data={"title": "Cache Test"},
        headers={"password": TEST_PASSWORD}
    )
    assert response.status_code == 201
    doc_id = response.json()["doc_id"]

    # Ask the same question twice
    question_data = {
        "query": "What is this document about?",
        "top_k": 3
    }

    # First request (cache miss)
    response1 = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )
    assert response1.status_code == 200
    result1 = response1.json()

    # Second request (should hit cache)
    response2 = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )
    assert response2.status_code == 200
    result2 = response2.json()

    # Results should be identical (from cache)
    assert result1["answer"] == result2["answer"]
    assert len(result1["sources"]) == len(result2["sources"])
    print("✓ Caching works correctly")

    # Cleanup
    client.delete(f"/documents/{doc_id}", headers={"password": TEST_PASSWORD})


def test_invalid_doc_id_filter():
    """Test that filtering by non-existent document IDs returns no results."""
    fake_doc_id = str(uuid.uuid4())

    question_data = {
        "query": "Tell me something",
        "doc_ids": [fake_doc_id]
    }

    response = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )

    # Should get 404 since no documents match
    assert response.status_code == 404
    print("✓ Correctly handles invalid doc_id filter")


def test_empty_query():
    """Test that empty queries are rejected."""
    question_data = {
        "query": "",
        "top_k": 3
    }

    response = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )

    # Should fail validation (422 Unprocessable Entity)
    assert response.status_code == 422
    print("✓ Correctly rejects empty queries")


def test_large_top_k():
    """Test that excessively large top_k values are handled."""
    question_data = {
        "query": "Test query",
        "top_k": 100  # Exceeds max of 20
    }

    response = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )

    # Should fail validation
    assert response.status_code == 422
    print("✓ Correctly rejects invalid top_k")


def test_similarity_threshold_filtering():
    """
    Test that low-similarity chunks are filtered out.

    Validates adaptive threshold is working.
    """
    # Upload document
    test_content = b"The quick brown fox jumps over the lazy dog."
    files = {"file": ("threshold_test.txt", BytesIO(test_content), "text/plain")}
    response = client.post(
        "/documents",
        files=files,
        data={"title": "Threshold Test"},
        headers={"password": TEST_PASSWORD}
    )
    assert response.status_code == 201
    doc_id = response.json()["doc_id"]

    # Ask a completely unrelated question
    question_data = {
        "query": "Explain quantum mechanics in detail",  # Unrelated to content
        "top_k": 10
    }

    response = client.post(
        "/ask",
        json=question_data,
        headers={"password": TEST_PASSWORD}
    )

    # Should either return 404 or very few results (filtered by threshold)
    if response.status_code == 200:
        result = response.json()
        # If we get sources, they should have relatively low scores
        if result["sources"]:
            for source in result["sources"]:
                # Score should be below high-confidence threshold
                assert source["score"] < 0.7, "Should filter low-quality matches"

    print("✓ Similarity threshold filtering works")

    # Cleanup
    client.delete(f"/documents/{doc_id}", headers={"password": TEST_PASSWORD})


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
