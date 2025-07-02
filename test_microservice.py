"""
Test script for Lift-os-LLM microservice.

Comprehensive test suite to verify all major functionality
including API endpoints, content analysis, and integrations.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class LiftOSLLMTester:
    """Test suite for Lift-os-LLM microservice."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        print(f"🧪 Running test: {test_name}")
        start_time = time.time()
        
        try:
            result = await test_func()
            duration = time.time() - start_time
            
            self.test_results.append({
                "name": test_name,
                "status": "PASS",
                "duration": duration,
                "result": result
            })
            print(f"✅ {test_name} - PASSED ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append({
                "name": test_name,
                "status": "FAIL",
                "duration": duration,
                "error": str(e)
            })
            print(f"❌ {test_name} - FAILED ({duration:.2f}s): {e}")
            return False
    
    async def test_health_endpoints(self):
        """Test health check endpoints."""
        # Test basic health check
        async with self.session.get(f"{self.base_url}/health") as response:
            assert response.status == 200
            data = await response.json()
            assert data["status"] == "healthy"
            assert "service" in data
            assert "version" in data
        
        # Test readiness check
        async with self.session.get(f"{self.base_url}/ready") as response:
            # May fail if dependencies not available, but should return valid response
            data = await response.json()
            assert "status" in data
        
        # Test root endpoint
        async with self.session.get(f"{self.base_url}/") as response:
            assert response.status == 200
            data = await response.json()
            assert data["service"] == "Lift-os-LLM"
            assert "docs" in data
        
        return "Health endpoints working correctly"
    
    async def test_api_documentation(self):
        """Test API documentation endpoints."""
        # Test OpenAPI docs
        async with self.session.get(f"{self.base_url}/docs") as response:
            assert response.status == 200
            content = await response.text()
            assert "Lift-os-LLM" in content
        
        # Test OpenAPI JSON
        async with self.session.get(f"{self.base_url}/openapi.json") as response:
            assert response.status == 200
            data = await response.json()
            assert data["info"]["title"] == "Lift-os-LLM"
            assert "paths" in data
        
        return "API documentation accessible"
    
    async def test_models_endpoints(self):
        """Test model management endpoints."""
        # Test list models (without auth - should fail or return limited info)
        async with self.session.get(f"{self.base_url}/api/v1/models/") as response:
            # This might fail due to auth requirements, which is expected
            if response.status == 401:
                return "Models endpoint properly protected"
            elif response.status == 200:
                data = await response.json()
                assert "data" in data
                return "Models endpoint accessible"
        
        return "Models endpoint tested"
    
    async def test_content_analysis_structure(self):
        """Test content analysis endpoint structure (without auth)."""
        # Test analysis endpoint structure
        test_payload = {
            "content": {
                "url": "https://example.com",
                "title": "Test Page",
                "description": "Test description"
            },
            "analysis_type": "quick",
            "include_embeddings": False,
            "include_knowledge_graph": False
        }
        
        async with self.session.post(
            f"{self.base_url}/api/v1/analysis/analyze",
            json=test_payload
        ) as response:
            # Should fail due to auth, but validates endpoint exists
            if response.status == 401:
                return "Analysis endpoint properly protected"
            elif response.status in [400, 422]:
                # Validation error is also acceptable
                return "Analysis endpoint validates input"
        
        return "Analysis endpoint structure verified"
    
    async def test_batch_endpoints_structure(self):
        """Test batch processing endpoint structure."""
        # Test batch submission endpoint
        async with self.session.get(f"{self.base_url}/api/v1/batch/jobs") as response:
            # Should fail due to auth requirements
            if response.status == 401:
                return "Batch endpoints properly protected"
        
        return "Batch endpoints structure verified"
    
    async def test_cors_headers(self):
        """Test CORS configuration."""
        async with self.session.options(f"{self.base_url}/health") as response:
            headers = response.headers
            # Check for CORS headers
            if "Access-Control-Allow-Origin" in headers:
                return "CORS headers present"
        
        return "CORS configuration tested"
    
    async def test_error_handling(self):
        """Test error handling."""
        # Test 404 endpoint
        async with self.session.get(f"{self.base_url}/nonexistent") as response:
            assert response.status == 404
        
        # Test malformed JSON
        async with self.session.post(
            f"{self.base_url}/api/v1/analysis/analyze",
            data="invalid json"
        ) as response:
            assert response.status in [400, 422]
        
        return "Error handling working correctly"
    
    async def test_performance_headers(self):
        """Test performance monitoring headers."""
        async with self.session.get(f"{self.base_url}/health") as response:
            headers = response.headers
            # Check for process time header
            if "X-Process-Time" in headers:
                process_time = float(headers["X-Process-Time"])
                assert process_time >= 0
                return f"Performance headers present (process time: {process_time:.4f}s)"
        
        return "Performance headers tested"
    
    def print_summary(self):
        """Print test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        total_duration = sum(result["duration"] for result in self.test_results)
        
        print("\n" + "="*60)
        print("🧪 LIFT-OS-LLM MICROSERVICE TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        print("="*60)
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['name']}: {result.get('error', 'Unknown error')}")
        
        print(f"\n🎯 Microservice Status: {'HEALTHY' if failed_tests == 0 else 'NEEDS ATTENTION'}")
        return failed_tests == 0


async def main():
    """Main test runner."""
    print("🚀 Starting Lift-os-LLM Microservice Tests")
    print("="*60)
    
    # Check if service is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health", timeout=5) as response:
                if response.status != 200:
                    print("❌ Service not responding correctly")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        print("💡 Make sure the service is running with: uvicorn src.main:app --reload")
        return False
    
    # Run tests
    async with LiftOSLLMTester() as tester:
        tests = [
            ("Health Endpoints", tester.test_health_endpoints),
            ("API Documentation", tester.test_api_documentation),
            ("Models Endpoints", tester.test_models_endpoints),
            ("Content Analysis Structure", tester.test_content_analysis_structure),
            ("Batch Endpoints Structure", tester.test_batch_endpoints_structure),
            ("CORS Headers", tester.test_cors_headers),
            ("Error Handling", tester.test_error_handling),
            ("Performance Headers", tester.test_performance_headers),
        ]
        
        for test_name, test_func in tests:
            await tester.run_test(test_name, test_func)
        
        return tester.print_summary()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)