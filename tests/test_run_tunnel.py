import io
import unittest
from unittest.mock import patch

from backend.run_tunnel import tunnel_url_from_log, wait_for_health


class RunningProcess:
    def poll(self) -> None:
        return None


class TunnelLogTests(unittest.TestCase):
    def test_extracts_quick_tunnel_url(self) -> None:
        self.assertEqual(
            tunnel_url_from_log("INF | https://quiet-bird.trycloudflare.com |"),
            "https://quiet-bird.trycloudflare.com",
        )

    def test_ignores_cloudflare_api_url(self) -> None:
        self.assertIsNone(
            tunnel_url_from_log("INF Requesting a tunnel from https://api.trycloudflare.com")
        )

    @patch("backend.run_tunnel.urllib.request.urlopen")
    def test_health_check_accepts_json_whitespace(self, urlopen: unittest.mock.Mock) -> None:
        response = io.BytesIO(b'{ "status": "ok" }\n')
        response.status = 200
        urlopen.return_value.__enter__.return_value = response

        self.assertIsNone(wait_for_health("http://127.0.0.1:8000/health", 1, RunningProcess()))


if __name__ == "__main__":
    unittest.main()
