import http from "k6/http";
import { check, sleep } from "k6";

const targetRate = Number(__ENV.RATE || 100);
const duration = __ENV.DURATION || "1m";
const baseUrl = __ENV.BASE_URL || "http://localhost:8000";
const preAllocatedVUs = Number(__ENV.PREALLOCATED_VUS || 50);
const maxVUs = Number(__ENV.MAX_VUS || 300);

export const options = {
  scenarios: {
    steady_load: {
      executor: "constant-arrival-rate",
      rate: targetRate,
      timeUnit: "1s",
      duration,
      preAllocatedVUs,
      maxVUs,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<200"],
  },
};

const payload = JSON.stringify({
  mode: "text",
  text: "Explain how retrieval grounding improves factuality.",
  top_k: 5,
  max_context_chunks: 6,
  fast_mode: true,
});

const params = {
  headers: {
    "Content-Type": "application/json",
  },
};

export default function () {
  const res = http.post(`${baseUrl}/v1/query`, payload, params);
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
  sleep(0.01);
}
