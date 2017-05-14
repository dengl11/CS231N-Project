import glob
import imageio

def compile_frames_to_gif(images, gif_file, duration=0.1):
	gif = imageio.mimsave(gif_file, images, duration=duration)
	return gif

