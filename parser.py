import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

# Load the datasets
df1 = pd.read_csv('path_to_speed_vs_time.csv')  # Speed vs. time dataset
df2 = pd.read_csv('path_to_event_metadata.csv')  # Metadata (RecordingId, categories)

# Convert time columns to datetime for filtering
df1['_time'] = pd.to_datetime(df1['_time'])
df2['StartTime'] = pd.to_datetime(df2['StartTime'])
df2['EndTime'] = pd.to_datetime(df2['EndTime'])

# Function to generate speed vs time plot and save as image
def generate_speed_plot(recording_id, save_dir='plots', dpi=32):
    """
    This function generates and saves a speed vs time plot for a specific recording ID.
    It saves the plot as a 64x64 image, with a classification label in the filename.
    
    Args:
    - recording_id: str, the unique identifier of the recording (from metadata).
    - save_dir: str, directory to save the images.
    - dpi: int, resolution of the saved plot, default is 32 DPI for 64x64 images.
    """
    try:
        # Fetch the event data for the given recording ID
        event_data = df2[df2['RecordingId'] == recording_id]
        if event_data.empty:
            raise ValueError(f"Recording ID {recording_id} not found in metadata.")
        # Extract event details
        event_data = event_data.iloc[0]
        serial = event_data['Serial']
        start_time = event_data['StartTime']
        end_time = event_data['EndTime']
        is_traffic_stop = event_data['TrafficStop']  # 1 for Traffic Stop, 0 for Non-Traffic Stop

        # Filter speed data for the relevant time range and serial
        speed_data = df1[(df1['host'] == serial) & 
                         (df1['_time'] >= start_time) & 
                         (df1['_time'] <= end_time)]
        if speed_data.empty:
            raise ValueError(f"No speed data available for Recording ID {recording_id}.")

        # Create a 64x64 pixel plot
        plt.figure(figsize=(2, 2))  # 2x2 inches at 32 DPI gives 64x64 pixels
        plt.plot(speed_data['_time'], speed_data['Speed'], color='blue', linewidth=1)
        plt.title('Speed vs. Time')
        plt.xlabel('Time')
        plt.ylabel('Speed')

        # Format x-axis to display time properly
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

        # Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Assign label and filename based on Traffic Stop classification
        #label = 'Traffic_Stop' if !(is_traffic_stop) == 1 else 'Non_Traffic_Stop'
        #filename = f'{recording_id}_{label}.png'
        #image_path = os.path.join(save_dir, filename)

        event_data = event_data.iloc[0]
        serial = event_data['Serial']
        if speed_data.empty:
            raise ValueError(f"No speed data available for Recording ID {recording_id}")

        # Save the plot as a 64x64 image
        plt.savefig(image_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved plot for {recording_id} as {image_path}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# Function to generate plots for multiple recordings
def generate_multiple_plots(recording_ids, save_dir='plots', dpi=32):
    """
    Generate speed vs time plots for a list of recording IDs and save them as 64x64 images.
    
    Args:
    - recording_ids: list, list of recording IDs to process.
    - save_dir: str, directory to save the images.
    - dpi: int, resolution of the saved plot, default is 32 DPI for 64x64 images.
    """
    for recording_id in recording_ids:
        generate_speed_plot(recording_id, save_dir, dpi)


# Example usage: Generate plot for a specific recording ID
recording_id_example = '66de3bfd-08fc-d720-4afc-215b944d681b'  # Replace with actual Recording ID
generate_speed_plot(recording_id_example)

# Example usage: Generate plots for multiple recordings
recording_ids_list = ['66de3bfd-08fc-d720-4afc-215b944d681b', 
                      '66de3c78-f0f7-cc98-f194-215b944ed040']  # Replace with actual IDs
generate_multiple_plots(recording_ids_list)
