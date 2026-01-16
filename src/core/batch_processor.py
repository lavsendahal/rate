"""Improved batch processing with better error handling and validation."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from .exceptions import BatchProcessingError
from .validators import ResultValidator


class BatchProcessor:
    """Handles batch processing with improved error handling and validation."""
    
    def __init__(self, client, config: Dict, logger, validator: ResultValidator):
        self.client = client
        self.config = config
        self.logger = logger
        self.validator = validator
        self.max_retries = config.get('processing', {}).get('max_retries', 2)
        self.retry_delay = config.get('processing', {}).get('retry_delay', 5)
        self.max_concurrency = int(config.get('processing', {}).get('max_concurrency', 8))
        self._http_timeout_s = float(config.get("processing", {}).get("request_timeout_s", 120))
        self._server_base_url = f"{config['server']['base_url']}:{config['server']['port']}"
        self.logger.info(
            f"BatchProcessor settings: max_concurrency={self.max_concurrency}, "
            f"max_retries={self.max_retries}, retry_delay={self.retry_delay}s"
        )
    
    def process_batch(self, requests: List[Tuple[str, str]], stage_name: str) -> Dict[str, str]:
        """Process a batch of requests with improved error handling and validation.

        Notes:
            Some OpenAI-compatible servers (including common sglang setups) do not implement the
            OpenAI Batch API endpoints (`/v1/files`, `/v1/batches`). In that case, we fall back to
            issuing normal `/v1/chat/completions` requests concurrently.
        """
        if not requests:
            return {}
        
        self.logger.info(f"Processing batch of {len(requests)} requests for {stage_name}")

        # Fast-fail: if batch endpoints are not supported, retries won't help.
        try_batch_api = True

        for attempt in range(self.max_retries + 1):
            try:
                if not try_batch_api:
                    raise BatchProcessingError("Batch API disabled; using direct chat completions fallback")

                # Submit batch and wait for completion
                batch_response = self._submit_and_wait_for_batch(requests, attempt)
                
                # Extract and validate results
                results = self._extract_batch_results(batch_response, requests, stage_name)
                
                # Validate results
                validated_results = self.validator.validate_batch_results(results, requests, stage_name)
                
                self.logger.info(f"Successfully processed {len(validated_results)}/{len(requests)} requests")
                return validated_results
                
            except BatchProcessingError as e:
                self.logger.warning(f"Batch processing attempt {attempt + 1} failed: {str(e)}")
                if self._looks_like_not_found(e):
                    # The server is reachable but doesn't implement batch endpoints.
                    self.logger.warning(
                        "OpenAI Batch API endpoints not supported by server (404). "
                        "Falling back to direct /v1/chat/completions requests."
                    )
                    try_batch_api = False
                    break
                if attempt < self.max_retries:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed for batch")
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in batch processing: {str(e)}")
                raise BatchProcessingError(f"Unexpected batch processing error: {str(e)}")

        # Fallback path: normal chat completions.
        results = self._process_requests_via_chat_completions(requests)
        validated_results = self.validator.validate_batch_results(results, requests, stage_name)
        self.logger.info(
            f"Successfully processed {len(validated_results)}/{len(requests)} requests via chat completions fallback"
        )
        return validated_results

    def _looks_like_not_found(self, err: Exception) -> bool:
        msg = str(err)
        return "404" in msg and ("Not Found" in msg or "not found" in msg)

    def _process_requests_via_chat_completions(self, requests: List[Tuple[str, str]]) -> Dict[str, str]:
        """Issue individual chat completion requests (with limited concurrency).

        We use direct HTTP calls instead of the OpenAI Python SDK here to be robust to
        minor schema differences across OpenAI-compatible servers.
        """
        if not requests:
            return {}

        model_cfg = self.config.get("model", {})
        model_name = model_cfg.get("name")
        if not model_name:
            raise BatchProcessingError("Missing config.model.name for chat completions fallback")

        temperature = model_cfg.get("temperature", 0.1)
        top_p = model_cfg.get("top_p", 0.1)
        max_tokens = model_cfg.get("max_tokens", 4096)

        url = f"{self._server_base_url}/v1/chat/completions"

        def extract_content(resp_obj: Dict[str, Any]) -> str:
            # Error payloads
            if isinstance(resp_obj.get("error"), dict):
                return ""

            # OpenAI schema: choices: [{message:{content:"..."}}]
            # Some servers: choices is an object not a list.
            choices = resp_obj.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0] or {}
                msg = first.get("message") or {}
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg.get("content") or "")
                # Alternative schemas
                if "text" in first:
                    return str(first.get("text") or "")
                delta = first.get("delta") or {}
                if isinstance(delta, dict) and "content" in delta:
                    return str(delta.get("content") or "")
                return ""
            if isinstance(choices, dict):
                msg = choices.get("message") or {}
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg.get("content") or "")
                if "text" in choices:
                    return str(choices.get("text") or "")
                return ""
            # Some servers put the output at top level.
            if "content" in resp_obj:
                return str(resp_obj.get("content") or "")
            if "text" in resp_obj:
                return str(resp_obj.get("text") or "")
            return ""

        def run_one(request_id: str, prompt: str) -> Tuple[str, str]:
            try:
                body = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Please output in plain text without any formatting.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                req = Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=self._http_timeout_s) as r:
                    raw = r.read().decode("utf-8")
                resp_obj = json.loads(raw) if raw else {}
                if isinstance(resp_obj.get("error"), dict):
                    self.logger.warning(f"Chat completion error payload for {request_id}: {resp_obj.get('error')}")
                content = extract_content(resp_obj)
                if not content and raw:
                    # Log a small sample once per batch_processor instance to help debug schema mismatches.
                    if not hasattr(self, "_logged_empty_response_sample"):
                        setattr(self, "_logged_empty_response_sample", True)
                        self.logger.warning(
                            "Chat completion returned no parseable content; sample payload: "
                            + raw[:500].replace("\n", " ")
                        )
                if content and "</think>" in content:
                    content = content.split("</think>")[-1]
                return request_id, content.strip()
            except HTTPError as e:
                # If the server doesn't support /v1/chat/completions, surface it loudly.
                try:
                    detail = e.read().decode("utf-8")
                except Exception:
                    detail = str(e)
                self.logger.warning(f"Chat completion HTTP error for {request_id}: {e.code} {detail}")
                return request_id, ""
            except URLError as e:
                self.logger.warning(f"Chat completion connection error for {request_id}: {e}")
                return request_id, ""
            except Exception as e:
                # Keep behavior consistent with batch mode: missing/failed requests become empty strings.
                self.logger.warning(f"Chat completion failed for {request_id}: {e}")
                return request_id, ""

        max_workers = max(1, min(self.max_concurrency, len(requests)))
        results: Dict[str, str] = {}

        if max_workers == 1:
            for request_id, prompt in requests:
                rid, content = run_one(request_id, prompt)
                results[rid] = content
            return results

        # Concurrency helps throughput against local servers.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, request_id, prompt) for request_id, prompt in requests]
            for fut in as_completed(futures):
                rid, content = fut.result()
                results[rid] = content
        return results
    
    def _submit_and_wait_for_batch(self, requests: List[Tuple[str, str]], attempt: int) -> Any:
        """Submit batch and wait for completion."""
        # Create batch requests
        batch_requests = [
            {
                "custom_id": request_id,
                "method": "POST", 
                "url": "/chat/completions",
                "body": {
                    "model": self.config['model']['name'],
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Please output in plain text without any formatting."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config['model']['temperature'],
                    "top_p": self.config['model']['top_p'],
                    "max_tokens": self.config['model']['max_tokens']
                }
            }
            for request_id, prompt in requests
        ]
        
        # Create and submit batch file
        temp_file = None
        try:
            temp_file = self._create_batch_file(batch_requests, attempt)
            file_response = self._upload_batch_file(temp_file)
            batch_response = self._create_batch_request(file_response.id)
            
            # Wait for completion with progress tracking
            return self._wait_for_batch_completion(batch_response)
            
        finally:
            # Clean up temporary file
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete temp file {temp_file}: {e}")
    
    def _create_batch_file(self, batch_requests: List[Dict], attempt: int) -> Path:
        """Create temporary batch file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = Path(f"temp_batch_{timestamp}_attempt_{attempt}.jsonl")
        
        try:
            with open(temp_file, "w") as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")
            return temp_file
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise BatchProcessingError(f"Failed to create batch file: {str(e)}")
    
    def _upload_batch_file(self, temp_file: Path) -> Any:
        """Upload batch file to API."""
        try:
            with open(temp_file, "rb") as f:
                file_response = self.client.files.create(file=f, purpose="batch")
            return file_response
        except Exception as e:
            raise BatchProcessingError(f"Failed to upload batch file: {str(e)}")
    
    def _create_batch_request(self, file_id: str) -> Any:
        """Create batch processing request."""
        try:
            batch_response = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            return batch_response
        except Exception as e:
            raise BatchProcessingError(f"Failed to create batch request: {str(e)}")
    
    def _wait_for_batch_completion(self, batch_response: Any) -> Any:
        """Wait for batch completion with progress tracking."""
        with tqdm(desc=f"Waiting for batch {batch_response.id[:8]}...", 
                 unit="check", 
                 bar_format='{desc}: {elapsed}') as pbar:
            
            while batch_response.status not in ["completed", "failed", "cancelled"]:
                time.sleep(3)
                try:
                    batch_response = self.client.batches.retrieve(batch_response.id)
                    pbar.update(1)
                    pbar.set_description(f"Batch {batch_response.id[:8]} - {batch_response.status}")
                except Exception as e:
                    raise BatchProcessingError(f"Failed to check batch status: {str(e)}")
        
        if batch_response.status != "completed":
            raise BatchProcessingError(f"Batch failed with status: {batch_response.status}")
        
        return batch_response
    
    def _extract_batch_results(self, batch_response: Any, requests: List[Tuple[str, str]], stage_name: str) -> Dict[str, str]:
        """Extract results from completed batch."""
        try:
            # Download results
            result_content = self.client.files.content(batch_response.output_file_id).read().decode("utf-8")
            
            # Log the first few lines for debugging
            lines = result_content.split("\n")
            self.logger.debug(f"Batch response has {len([l for l in lines if l.strip()])} lines")
            if lines and lines[0].strip():
                self.logger.debug(f"First response line: {lines[0][:200]}...")
            
            # Parse results
            results = {}
            for line_num, line in enumerate(lines):
                if line.strip():
                    try:
                        result = json.loads(line)
                        custom_id = result['custom_id']
                        
                        # Try parsing the sglang response format
                        content = ""
                        parsing_success = False
                        
                        # Log full structure for first result to help with debugging
                        if line_num == 0:
                            self.logger.debug(f"First result structure: {json.dumps(result, indent=2)}")
                        
                        # sglang format: choices is an object, not an array
                        if 'response' in result and 'body' in result['response']:
                            try:
                                choices = result['response']['body']['choices']
                                # Handle both object and array formats
                                if isinstance(choices, dict):
                                    content = choices['message']['content']
                                elif isinstance(choices, list):
                                    content = choices[0]['message']['content']
                                parsing_success = True
                                if line_num == 0:
                                    self.logger.debug("Successfully parsed sglang response format")
                            except (KeyError, IndexError, TypeError) as e:
                                self.logger.debug(f"Failed to parse expected format: {e}")
                        
                        # Handle error responses
                        if not parsing_success and 'error' in result:
                            self.logger.warning(f"Error in batch response for {custom_id}: {result['error']}")
                            content = ""
                            parsing_success = True
                        
                        if not parsing_success:
                            self.logger.warning(f"Could not parse response for {custom_id}, available keys: {list(result.keys())}")
                            content = ""
                        
                        # Clean up content (remove thinking tags if present)
                        if content and "</think>" in content:
                            content = content.split("</think>")[-1]
                        
                        results[custom_id] = content.strip() if content else ""
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Could not parse result line {line_num}: {line[:100]}... Error: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Unexpected error parsing line {line_num}: {e}")
                        continue
            
            # Clean up API files
            try:
                self.client.files.delete(batch_response.output_file_id)
                self.client.files.delete(batch_response.input_file_id)
            except Exception as e:
                self.logger.warning(f"Could not delete API files: {e}")
            
            # Check if we got results for all requests
            request_ids = {req_id for req_id, _ in requests}
            missing_ids = request_ids - set(results.keys())
            
            if missing_ids:
                self.logger.warning(f"Missing results for {len(missing_ids)} requests: {list(missing_ids)[:5]}...")
                # Add empty results for missing IDs to maintain consistency
                for missing_id in missing_ids:
                    results[missing_id] = ""
            
            # Log summary with detailed breakdown
            non_empty_results = sum(1 for v in results.values() if v.strip())
            empty_results = len(results) - non_empty_results
            self.logger.info(f"Extracted {len(results)} results: {non_empty_results} non-empty, {empty_results} empty")
            
            # Log sample results for debugging - include some empty ones
            sample_results = list(results.items())[:5]
            empty_samples = [(k, v) for k, v in results.items() if not v.strip()][:3]
            
            self.logger.info(f"Sample non-empty results:")
            for rid, content in sample_results:
                if content.strip():
                    self.logger.info(f"  {rid}: '{content[:150]}{'...' if len(content) > 150 else ''}'")
            
            if empty_samples:
                self.logger.warning(f"Sample empty results:")
                for rid, content in empty_samples:
                    self.logger.warning(f"  {rid}: EMPTY ('{content}')")
            
            # Log response patterns to identify issues
            response_patterns = {}
            for content in results.values():
                content_lower = content.strip().lower()
                if not content_lower:
                    key = "EMPTY"
                elif "no relevant findings" in content_lower:
                    key = "no_relevant_findings"
                elif "no findings" in content_lower:
                    key = "no_findings"  
                elif "normal" in content_lower:
                    key = "normal"
                elif len(content_lower) < 10:
                    key = "very_short"
                else:
                    key = "substantial"
                
                response_patterns[key] = response_patterns.get(key, 0) + 1
            
            self.logger.info(f"Response patterns: {response_patterns}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to extract batch results: {str(e)}")
            # Include more details in the error for debugging
            self.logger.error(f"Batch response status: {getattr(batch_response, 'status', 'unknown')}")
            self.logger.error(f"Batch response ID: {getattr(batch_response, 'id', 'unknown')}")
            raise BatchProcessingError(f"Failed to extract batch results: {str(e)}")
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        } 
