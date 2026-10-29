"""Unit and smoke tests for the House Price Prediction Flask API."""
import unittest
import json
from app import app

class FlaskAPITestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))

    def test_metrics_endpoint(self):
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('r2_score', data.get('metrics', {}))
        self.assertAlmostEqual(data['metrics']['r2_score'], 0.7944, places=2)

    def test_predict_endpoint_success(self):
        payload = {
            "OverallQual": 6,
            "GrLivArea": 1500,
            "TotalBsmtSF": 1000,
            "GarageCars": 2,
            "FullBath": 2,
            "YearBuilt": 2000
        }
        response = self.client.post('/api/predict',
                                    data=json.dumps(payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('predicted_price', data)
        self.assertGreater(data['predicted_price'], 0)
        print("\nTest prediction output:", data['formatted_price'])

    def test_predict_invalid_input(self):
        response = self.client.post('/api/predict',
                                    data="invalid string",
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
