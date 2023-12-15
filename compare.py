import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def compare_results(input_file1="paper_result.png", input_file2="./results/my_result.png", output_file="./results/final_result.png"):
    """
    Compare two images and create a side-by-side plot.

    Parameters:
    - input_file1 (str): Path to the first input image file. Default is "paper_result.png".
    - input_file2 (str): Path to the second input image file. Default is "./results/my_result.png".
    - output_file (str): Path to save the output comparison plot. Default is "./results/final_result.png".

    Returns:
    None

    Example:
    compare_results(input_file1="paper_result.png", input_file2="./results/my_result.png", output_file="./results/final_result.png")
    """
    # Read the images using Matplotlib
    image1 = mpimg.imread(input_file1)
    image2 = mpimg.imread(input_file2)

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1)

    # Plot the first image on top
    axes[0].imshow(image1)
    axes[0].axis('off')

    # Plot the second image on bottom
    axes[1].imshow(image2)
    axes[1].axis('off')

    # Save the plot
    plt.savefig(output_file)


# Call the function
compare_results(input_file1="paper_result.png", input_file2="./results/my_result.png", output_file="./results/final_result.png")
