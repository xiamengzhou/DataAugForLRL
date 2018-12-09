import sys

dict_file = sys.argv[1]

format_dict =  sys.argv[2]

with open(dict_file, encoding='utf-8') as f1, open(format_dict, 'w', encoding='utf-8') as f2:
	for line in f1:
		words = line.strip().split('|||')
		f2.write(words[0].strip() + ' ' + words[1].strip() + '\n')
