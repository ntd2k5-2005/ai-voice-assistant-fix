#!/usr/bin/env python3
"""
Ollama Model Setup Script
Simple script to check GPU and pull Qwen model

Usage:
    python setup_ollama_model.py
    python setup_ollama_model.py --model qwen3.5:4b
"""

import subprocess
import time
import sys
import argparse


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_gpu():
    """Check GPU availability"""
    print_header("GPU INFORMATION")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            print(f" GPU: {gpu_info[0]}")
            print(f" Total VRAM: {gpu_info[1]}")
            print(f" Free VRAM: {gpu_info[2]}")
            print(" GPU is available for Ollama")
            return True
        else:
            print("  No GPU detected")
            print(" CPU mode will be used (slower)")
            return False
            
    except FileNotFoundError:
        print("  nvidia-smi not found - no NVIDIA GPU detected")
        print(" CPU mode will be used (slower)")
        return False
        
    except subprocess.TimeoutExpired:
        print("  GPU check timed out")
        return False
        
    except Exception as e:
        print(f" Error checking GPU: {e}")
        return False


def check_ollama():
    """Check if Ollama is installed and running"""
    print_header("OLLAMA STATUS")
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f" Ollama installed: {version}")
        else:
            print(" Ollama not found")
            print("\nPlease install Ollama:")
            print("  Windows: https://ollama.com/download")
            print("  Linux:   curl -fsSL https://ollama.com/install.sh | sh")
            print("  macOS:   brew install ollama")
            return False
            
    except FileNotFoundError:
        print(" Ollama not found")
        print("\nPlease install Ollama:")
        print("  Windows: https://ollama.com/download")
        print("  Linux:   curl -fsSL https://ollama.com/install.sh | sh")
        print("  macOS:   brew install ollama")
        return False
        
    except Exception as e:
        print(f" Error checking Ollama: {e}")
        return False
    
    # Check if Ollama server is running
    print("\n Checking Ollama server...")
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        
        if response.status_code == 200:
            print(" Ollama server is running")
            return True
        else:
            print("  Ollama server not responding properly")
            print(" Please start Ollama: ollama serve")
            return False
            
    except ImportError:
        print("  'requests' module not found")
        print("   Installing requests...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'requests', '-q'])
        print("   Please run this script again")
        return False
        
    except Exception:
        print("  Ollama server not running")
        print(" Please start Ollama in another terminal:")
        print("   ollama serve")
        return False


def list_models():
    """List currently installed models"""
    print_header("INSTALLED MODELS")
    
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                print(output)
            else:
                print("No models installed yet")
            return True
        else:
            print(" Failed to list models")
            return False
            
    except Exception as e:
        print(f" Error listing models: {e}")
        return False


def pull_model(model_name):
    """Pull Ollama model"""
    print_header(f"PULLING MODEL: {model_name}")
    
    # Check if model already exists
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if model_name in result.stdout:
            print(f"✅ Model {model_name} is already downloaded")
            
            # Ask if user wants to update
            response = input("\n🔄 Do you want to update it? (y/N): ").strip().lower()
            if response != 'y':
                print("Skipping download")
                return True
            
    except Exception:
        pass
    
    # Pull the model
    print(f"\n📥 Downloading model: {model_name}")
    print("⏳ This may take several minutes depending on model size...")
    print("💡 You can check progress below:\n")
    
    start_time = time.time()
    
    try:
        # Run ollama pull and show real-time output
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        
        if process.returncode == 0:
            elapsed = time.time() - start_time
            print(f"\n✅ Model {model_name} downloaded successfully!")
            print(f"⏱  Download time: {elapsed:.1f} seconds")
            return True
        else:
            print(f"\n Failed to download model {model_name}")
            return False
            
    except KeyboardInterrupt:
        print("\n\n  Download interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n Error downloading model: {e}")
        return False


def show_model_info(model_name):
    """Show model information"""
    print_header(f"MODEL INFO: {model_name}")
    
    try:
        result = subprocess.run(
            ['ollama', 'show', model_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f" Model {model_name} not found")
            return False
            
    except Exception as e:
        print(f" Error getting model info: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Setup Ollama and pull Qwen model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_ollama_model.py                    # Pull default qwen3.5:4b
  python setup_ollama_model.py --model qwen3.5:4b # Pull specific model
  python setup_ollama_model.py --list-only        # Just list installed models
        """
    )
    
    parser.add_argument(
        '--model',
        default='qwen3.5:4b',
        help='Model to pull (default: qwen3.5:4b)'
    )
    
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list installed models without pulling'
    )
    
    parser.add_argument(
        '--skip-gpu-check',
        action='store_true',
        help='Skip GPU check'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("   OLLAMA MODEL SETUP")
    print("="*70)
    
    # Check GPU
    if not args.skip_gpu_check:
        gpu_available = check_gpu()
    else:
        gpu_available = None
    
    # Check Ollama
    if not check_ollama():
        print("\n Setup failed: Ollama not available")
        print("\nPlease:")
        print("  1. Install Ollama from https://ollama.com")
        print("  2. Start Ollama server: ollama serve")
        print("  3. Run this script again")
        sys.exit(1)
    
    # List models
    list_models()
    
    # Pull model if not list-only
    if not args.list_only:
        print()
        success = pull_model(args.model)
        
        if success:
            print()
            show_model_info(args.model)
            
            print_header(" SETUP COMPLETE")
            print(f"Model '{args.model}' is ready to use!")
            print(f"\nNext steps:")
            print(f"  1. Test the model:")
            print(f"     ollama run {args.model}")
            print(f"  2. Or use it via API:")
            print(f"     python llm_api_standalone.py")
            print(f"     # or")
            print(f"     cd LLM-Model && python -m api.main")
        else:
            print_header(" SETUP FAILED")
            print("Please check the error messages above")
            sys.exit(1)
    else:
        print_header("✅ LIST COMPLETE")
        print("Use --model to pull a specific model")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)
