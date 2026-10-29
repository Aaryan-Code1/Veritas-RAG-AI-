import unittest

try:
    from fastapi.testclient import TestClient
    from src.api.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI or test dependencies not installed in current environment.")
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertEqual(data["service"], "veritas-ai-rag")

    def test_ingest_and_query_flow(self):
        # 1. Ingest document
        ingest_payload = {
            "title": "API Test Specification",
            "source_system": "confluence",
            "source_url": "https://wiki.test/api-spec",
            "content": "# Compliance\nCustomer data retention under GDPR is 30 days maximum.",
            "access_control_list": ["group:test-users"],
        }
        ingest_res = self.client.post("/api/v1/ingest", json=ingest_payload)
        self.assertEqual(ingest_res.status_code, 200)
        ingest_data = ingest_res.json()
        self.assertEqual(ingest_data["status"], "INDEXED")
        self.assertGreaterEqual(ingest_data["chunks_indexed"], 1)

        # 2. Query with matching user group
        query_payload = {
            "query": "What is customer data retention under GDPR?",
            "user_groups": ["group:test-users"],
            "stream": False,
        }
        query_res = self.client.post("/api/v1/query", json=query_payload)
        self.assertEqual(query_res.status_code, 200)
        query_data = query_res.json()
        self.assertIn("query_id", query_data)
        self.assertFalse(query_data["refusal"])
        self.assertGreaterEqual(len(query_data["citations"]), 1)

        # 3. Query with unauthorized user group (RBAC test)
        unauth_query_payload = {
            "query": "What is customer data retention under GDPR?",
            "user_groups": ["group:unauthorized-stranger"],
            "stream": False,
        }
        unauth_res = self.client.post("/api/v1/query", json=unauth_query_payload)
        self.assertEqual(unauth_res.status_code, 200)
        unauth_data = unauth_res.json()
        self.assertTrue(unauth_data["refusal"])


if __name__ == "__main__":
    unittest.main()
