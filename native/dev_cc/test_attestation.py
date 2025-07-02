#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 HyperBEAM Contributors. All rights reserved.
#
"""
Test script for HyperBEAM dev_cc_nif attestation service.
Tests generate and verify operations with multiple test cases.
"""

import subprocess
import json
import sys
import time
from typing import Dict, Any, Optional


class AttestationTester:
    """Test harness for attestation service."""
    
    def __init__(self, main_script_path: str = "main.py", verbose: bool = False):
        self.main_script_path = main_script_path
        self.test_count = 0
        self.passed_count = 0
        
    def _call_service(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the attestation service with given request data."""
        try:
            # Convert request to JSON
            json_input = json.dumps(request_data)
            
            # Call the main script
            process = subprocess.run(
                [sys.executable, self.main_script_path],
                input=json_input,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            # Parse response
            if process.returncode != 0:
                error_msg = f"Process failed with code {process.returncode}"
                if process.stderr:
                    error_msg += f": {process.stderr}"
                return {
                    "status": "error", 
                    "error": error_msg
                }
            
            try:
                # Try to parse JSON from stdout
                json_response = json.loads(process.stdout)
                return json_response
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "error": f"Invalid JSON response: {process.stdout}"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Service call timed out"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Service call failed: {str(e)}"
            }

    def _generate_attestation(self, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate an attestation token."""
        if test_data is None:
            test_data = {
                "name": "test-hyperbeam-node",
                "claims_version": "3.0",
                "device_type": "gpu",
                "environment": "local",
                "nonce": "da4a06c3604a5fac8aa0b4aaf5a6354cdd0dc7c193299bc3464f30b5cbfb931a"
            }

        request = {
            "action": "generate",
            "data": test_data
        }

        return self._call_service(request)

    def _verify_attestation(self, token: str) -> Dict[str, Any]:
        """Verify an attestation token."""
        request = {
            "action": "verify",
            "data": {
                "token": token,
                "name": "test-hyperbeam-node",
                "nonce": "da4a06c3604a5fac8aa0b4aaf5a6354cdd0dc7c193299bc3464f30b5cbfb931a"
            }
        }

        return self._call_service(request)

    def _log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        self.test_count += 1
        if success:
            self.passed_count += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"

        print(f"Test {self.test_count}: {test_name} - {status}")
        if details:
            print(f"  Details: {details}")
        print()

    def test_case_1(self) -> bool:
        """Test Case 1: Generate then Verify"""
        print("=" * 60)
        print("Test Case 1: Generate Attestation -> Verify Attestation")
        print("=" * 60)

        # Step 1: Generate attestation
        print("Step 1: Generating attestation...")
        generate_result = self._generate_attestation()

        if generate_result.get("status") != "ok":
            self._log_test_result(
                "Generate Attestation",
                False,
                f"Generation failed: {generate_result.get('error', 'Unknown error')}"
            )
            return False

        token = generate_result.get("result", {}).get("token")
        if not token:
            self._log_test_result(
                "Generate Attestation",
                False,
                "No token in generation result"
            )
            return False

        self._log_test_result("Generate Attestation", True,
                              f"Token length: {len(token)}")

        # Step 2: Verify attestation
        print("Step 2: Verifying attestation...")
        verify_result = self._verify_attestation(token)

        if verify_result.get("status") != "ok":
            self._log_test_result(
                "Verify Attestation",
                False,
                f"Verification failed: {verify_result.get('error', 'Unknown error')}"
            )
            return False

        is_valid = verify_result.get("result", {}).get("valid", False)
        self._log_test_result(
            "Verify Attestation",
            is_valid,
            f"Token validation result: {is_valid}"
        )

        return is_valid

    def test_case_2(self) -> bool:
        """Test Case 2: Execute Test Case 1 twice"""
        print("=" * 60)
        print("Test Case 2: Execute Test Case 1 Twice")
        print("=" * 60)

        success_count = 0

        # First execution
        print("First execution of Test Case 1:")
        if self.test_case_1():
            success_count += 1

        # Small delay between tests
        time.sleep(1)

        # Second execution
        print("Second execution of Test Case 1:")
        if self.test_case_1():
            success_count += 1

        overall_success = success_count == 2
        self._log_test_result(
            "Execute Test Case 1 Twice",
            overall_success,
            f"Successful executions: {success_count}/2"
        )

        return overall_success

    def run_all_tests(self):
        """Run all test cases."""

        # Run Test Case 1
        self.test_case_1()

        # Run Test Case 2
        self.test_case_2()


def main():
    """Main entry point for the test script."""
    if len(sys.argv) > 1:
        main_script = sys.argv[1]
    else:
        main_script = "main.py"

    tester = AttestationTester(main_script)

    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
