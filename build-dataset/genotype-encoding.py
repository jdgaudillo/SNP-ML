import pandas as pd
import numpy as np
import collections

"""" 
	This script represents SNP as 1, 2, 3 for homozygous, heterozygous, and
	variant homozygous, respectively.

	Input:
		file: dataset/preprocessed_matrix.csv
				Cleaned SNP dataset
	
	Output:
		file: dataset/encoded_snp.csv
				Encoded SNP dataset
""""

file = 'dataset/preprocessed_matrix.csv'
data = pd.read_csv(file, sep = ' ')
data = data.set_index('usersid')
snps = data.columns
snps = snps[:-1]
data = data[snps]
n_samples = 128

allele_freq = []
hetero = 0
ind_hetero = []
counter = 0

for snp in snps:
	print('SNP NUMBER:', counter)
	genotype_count = data[snp].value_counts().to_dict()
	genotypes = data[snp].unique()
	n_genotypes = len(genotypes)
	genotypes = sorted(genotypes)

	if n_genotypes > 3:
		het1 = genotype_count[genotypes[1]]
		het2 = genotype_count[genotypes[2]]
		hetero = het1 + het2
		genotype_count[genotypes[1]] = hetero
		del genotype_count[genotypes[2]]
		genotypes.remove(genotypes[2])

	for key in genotype_count:
		genotype_count[key] = genotype_count[key]/n_samples
	print('Observation', genotype_count)
	genotype_count = collections.OrderedDict(sorted(genotype_count.items()))

	if len(genotypes) == 3:
		allele_A = genotype_count[genotypes[0]] + (genotype_count[genotypes[1]] * 0.5)
		allele_B = genotype_count[genotypes[2]] + (genotype_count[genotypes[1]] * 0.5)
		print('allele:', allele_A)
		print('allele', allele_B)
		if allele_B > allele_A:
			encode = np.array([3,2,1])
		else:
			encode = np.array([1,2,3])
	elif len(genotypes) == 2:
		encode = np.array([1,2])
	else:
		encode = np.array([1])

	genotype_dict = dict(zip(genotypes, encode))
	data[snp] = data[snp].map(genotype_dict)

	print(data[snp].head())
	counter = counter + 1

outfile = 'encoded_snp.csv'
data.to_csv(outfile, sep = ' ')
