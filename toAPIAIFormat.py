def convert(input_file, output_file):
	with open(input_file) as f, open(output_file, "wb") as f2:
				for line in f:
					line = line.strip('\n')
					line = line.strip('\t')
					print line
					line = line.split("^")[0]
					output = '"%s","%s"\n' %(line, line)
					f2.write(output)

if __name__ == "__main__":
	convert('stockList.txt', 'stockListConverted.txt')