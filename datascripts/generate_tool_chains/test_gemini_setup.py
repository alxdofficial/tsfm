"""
Test Gemini Vertex AI setup with Application Default Credentials.

This script verifies that:
1. google-genai SDK is installed
2. ADC authentication is working
3. Vertex AI API is enabled
4. You can call Gemini models

Usage:
    python datascripts/generate_tool_chains/test_gemini_setup.py --project your-project-id
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Part
    print("✓ google-genai SDK installed")
except ImportError as e:
    print("✗ google-genai SDK not installed")
    print("  Install with: pip install google-genai")
    sys.exit(1)


def test_gemini_call(project_id: str, location: str = "us-central1"):
    """Test a simple Gemini API call."""
    print(f"\nTesting Gemini API call...")
    print(f"  Project: {project_id}")
    print(f"  Location: {location}")
    print(f"  Model: gemini-1.5-flash")

    try:
        # Initialize client
        client = genai.Client(
            vertexai=True,
            location=location,
            project=project_id
        )
        print("✓ Gemini client initialized")

        # Simple test prompt
        test_prompt = "Say 'Hello from Gemini!' and nothing else."

        # Configure
        config = GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=50,
        )

        # Call API
        print("\nCalling Gemini API...")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[{"role": "user", "parts": [Part(text=test_prompt)]}],
            config=config,
        )

        if response.text is None:
            print("✗ Response text is None")
            return False

        print(f"✓ Gemini response received:")
        print(f"  {response.text}")

        # Check for expected response
        if "Hello from Gemini" in response.text:
            print("\n✓ All tests passed!")
            return True
        else:
            print(f"\n⚠ Unexpected response: {response.text}")
            return True  # Still worked, just different response

    except Exception as e:
        print(f"\n✗ Error calling Gemini: {e}")
        print("\nCommon issues:")
        print("  1. ADC not set up: gcloud auth application-default login")
        print("  2. API not enabled: gcloud services enable aiplatform.googleapis.com")
        print("  3. Wrong project: gcloud config set project YOUR_PROJECT_ID")
        print("  4. No permissions: Ensure you have Vertex AI User role")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Gemini Vertex AI setup"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Google Cloud project ID"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="GCP region (default: us-central1)"
    )

    args = parser.parse_args()

    print("="*80)
    print("Gemini Vertex AI Setup Test")
    print("="*80)

    success = test_gemini_call(args.project, args.location)

    if success:
        print("\n✓ Setup is working correctly!")
        print("\nYou can now generate classification chains:")
        print(f"  python datascripts/generate_classification_chains.py \\")
        print(f"      --project {args.project} \\")
        print(f"      --dataset uci_har \\")
        print(f"      --num-samples 5")
        return 0
    else:
        print("\n✗ Setup test failed")
        print("  Please fix the issues above and try again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
