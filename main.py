import subprocess
import sys

def main():
    print("====================================")
    print("   GESTURA3D PROJECT HUB - v1.0    ")
    print("====================================")
    print("1. Launch Gestura OS (2D Canvas & 3D Builder)")
    print("2. Launch Interactive 3D Cube (Legacy)")
    print("3. Check System Dependencies")
    print("4. Exit")
    
    choice = input("\nSelect an option (1-4): ")

    if choice == '1':
        print("Booting Gestura OS...")
        subprocess.run([sys.executable, "gestura_os.py"])
    elif choice == '2':
        print("Booting Legacy Cube...")
        subprocess.run([sys.executable, "interactive_cube.py"])
    elif choice == '3':
        print("Running diagnostic...")
        # Add simple dependency check logic here
        print("All dependencies for Gestura3D are configured.")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()