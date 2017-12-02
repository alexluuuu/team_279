from utils.lbp import *

if __name__ == "__main__":
	print "it's me, the main boy"

	scale_adap_boy = SALBP()
	input_names = ['sample_dataset/ISIC_0000000.jpg']
	scale_adap_boy.ComputeLBP(input_names[0])
	scale_adap_boy.VisualizeLBP()