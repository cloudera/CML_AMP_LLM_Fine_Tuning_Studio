import os
import requests
from cairosvg import svg2png

# Define the icon dictionaries for each page
pages_icons = {
    "Fine Tuning Studio": {
        "Fine Tuning Studio": "architecture"
    },
    "Navigation": {
        "Home": "home",
    },
    "AI Toolkit": {
        "Import Datasets": "publish",
        "View Datasets": "data_object",
        "Import Base Models": "neurology",
        "View Base Models": "view_day",
        "Create Prompts": "chat",
        "View Prompts": "forum",
    },
    "Experiments": {
        "Train a New Adapter": "forward",
        "Monitor Training Jobs": "subscriptions",
        "Local Adapter Comparison": "difference",
        "Run MLFlow Evaluation": "model_training",
        "View MLflow Runs": "monitoring",
    },
    "CML": {
        "Export to CML Model Registry": "move_group",
        "Project Owner": "account_circle",
    }
}

# Define the base URL for downloading icons
base_url = "https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsoutlined/{}/default/48px.svg"

# Function to delete all files in the output directory
def clear_output_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# Function to download, color, and convert icons to PNG with improved resolution
def download_and_convert_icon(icon_name, color_hex, size, output_dir):
    icon_url = base_url.format(icon_name)
    response = requests.get(icon_url)
    if response.status_code == 200:
        svg_content = response.text

        # Inject fill and stroke color directly into the SVG path elements
        svg_content = svg_content.replace('<path ', f'<path fill="{color_hex}" stroke="{color_hex}" ')
        
        # Scale up SVG size for better resolution
        scaled_size = size * 4  # Increase scaling for better resolution
        output_png_path = os.path.join(output_dir, f"{icon_name}.png")
        
        # Convert SVG to PNG with the increased resolution
        svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_png_path, output_width=scaled_size, output_height=scaled_size)
        
        print(f"Downloaded and converted icon: {icon_name} in color {color_hex}")
    else:
        print(f"Failed to download icon: {icon_name}")

# Main function to process all icons
def process_icons(pages_icons, color_hex="#000000", size=48, output_dir="./downloaded_icons"):
    # Clear the output directory before processing
    clear_output_directory(output_dir)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for page, icons in pages_icons.items():
        for label, icon_name in icons.items():
            download_and_convert_icon(icon_name, color_hex, size, output_dir)

# Example usage
if __name__ == "__main__":
    # Set your desired color, size, and output directory here
    color_hex = "#3AA23A"  # Example: Green color
    icon_size = 25
    output_directory = "./resources/images/icons"

    # Process all icons
    process_icons(pages_icons, color_hex, icon_size, output_directory)

    print("All icons processed.")
