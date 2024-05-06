This readme.txt file was updated on 20221115 by Matthias Demuzere


-------------------
GENERAL INFORMATION
-------------------

Title of Dataset: Global map of Local Climate Zones

Principal Authors Information:

	Principal Investigator: Matthias Demuzere (matthias.demuzere ~at~ rub.de)
	Alternate Contact: Benjamin Bechtel (benjamin.bechtel ~at~ rub.de)

Summary: 
	A global 100 m spatial resolution Local Climate Zone (LCZ) map, derived from multiple earth 
	observation datasets and expert LCZ class labels. 
	
	The LCZ map is based on the LCZ typology (Stewart and Oke, 2012) that distinguish 
	urban surfaces accounting for their typical combination of micro-scale land-covers 
	and associated physical properties. The LCZ scheme is distinguished from other land use 
	/ land cover schemes by its focus on urban and rural landscape types, which can be 
	described by any of the 17 classes in the LCZ scheme. 

	Out of the 17 LCZ classes, 10 reflect the 'built' environment, and each LCZ type is 
	associated with generic numerical descriptions of key urban canopy parameters critical 
	to model atmospheric responses to urbanisation. In addition, since LCZs were originally 
	designed as a new framework for urban heat island studies (Stewart and Oke, 2012), they 
	also contain a limited set (7) of 'natural' land-cover classes that can be used as 'control' 
	or 'natural reference' areas. As these seven natural classes in the LCZ scheme can not 
	capture the heterogeneity of the world’s existing natural ecosystems, we advise 
	users - if required - to combine the built LCZ classes with any other land-cover product 
	that provides a wider range of natural land-cover classes.
	


--------------------------
SHARING/ACCESS INFORMATION
--------------------------

Licenses/restrictions placed on the data, or limitations of reuse: Creative Commons Attribution 4.0 International.

Recommended citations for the data:

	- Demuzere M, Kittner J, Martilli A, et al. A global map of local climate zones to support 
		earth system modelling and urban-scale environmental science. Earth Syst Sci Data. 
		2022a;14(8):3835-3873. doi:10.5194/essd-14-3835-2022
	- Demuzere M, Kittner J, Martilli A, et al. Global Local Climate Zone map. Zenodo (2022b)
		doi:10.5281/zenodo.6364594.

Other Notes:

	Permission is hereby granted, free of charge, to any person obtaining a copy of this data 
	and associated documentation files, to use these data without restriction, 
	subject to the following:

	- Understand that these data are created by crowd sourcing and machine learning and 
		thus will naturally contain some errors. It comes as it is without any warranty.
	- The above copyright notice and this permission notice shall be included in all copies or 
		substantial portions of the data.
	- The authors and WUDAPT contributors are acknowledged where appropriate
	- For scientific use, the following papers could be cited to introduce the LCZ concepts:
		* Stewart, I. D., & Oke, T. R. (2012). Local Climate Zones for Urban Temperature 
			Studies. Bulletin of the American Meteorological Society, 93(12), 1879–1900. 
			https://doi.org/10.1175/BAMS-D-11-00019.1
		* Bechtel B, Daneke C (2012) Classification of Local Climate Zones based on multiple 
			Earth Observation data. IEEE Journal of Selected Topics in Applied Earth 
			Observations and Remote Sensing 5:1191-1202
			http://doi.org/10.1109/JSTARS.2012.2189873
		* Bechtel B., Alexander P.J., Böhner J., Ching J., Conrad O., Feddema J., Mills G., 
			See L., Stewart I. (2015) Mapping Local Climate Zones for a Worldwide Database 
			of the Form and Function of Cities. ISPRS International Journal of 
			Geo-Information 4:199-219.
			http://doi.org/10.3390/ijgi4010199
		* Ching, J., Mills, G., Bechtel, B., See, L., Feddema, J., 
			Wang, X., … Theeuwes, N. (2018). WUDAPT: An Urban Weather, Climate, and 
			Environmental Modeling Infrastructure for the Anthropocene. 
			Bulletin of the American Meteorological Society, 99(9), 1907–1924. 
			https://doi.org/10.1175/BAMS-D-16-0236.1
		* Bechtel, B., Alexander, P. J., Beck, C., Böhner, J., Brousse, O., 
			Ching, J., … Xu, Y. (2019). Generating WUDAPT Level 0 data – Current status of 
			production and evaluation. Urban Climate, 27, 24–45. 
			https://doi.org/10.1016/j.uclim.2018.10.001
		* Demuzere, M., Bechtel, B., & Mills, G. (2019). Global transferability of 
			local climate zone models. Urban Climate, 27, 46–63. 
			https://doi.org/10.1016/j.uclim.2018.11.001
		* Demuzere, M., Bechtel, B., Middel, A., & Mills, G. (2019). Mapping Europe into 
			local climate zones. PLOS ONE, 14(4), e0214474. 
			https://doi.org/10.1371/journal.pone.0214474
		* Bechtel B., Demuzere M., Stewart ID. (2020) A Weighted Accuracy Measure for 
			Land Cover Mapping: Comment on Johnson et al. Local Climate Zone (LCZ) Map 
			Accuracy Assessments Should Account for Land Cover Physical Characteristics 
			that Affect the Local Thermal Environment. Remote Sens. 2019, 11, 2420. 
			Remote Sens. 12(11):1769. 
			https://doi.org/10.3390/rs12111769
		* Demuzere M., Hankey S., Mills G., Zhang W., Lu T., Bechtel B. (2020) Combining 
			expert and crowd-sourced training data to map urban form and functions 
			for the continental US. Sci Data. 7(1):264. 
			https://doi.org/10.1038/s41597-020-00605-z
		* Demuzere M., Kittner J., Bechtel B. (2021) LCZ Generator: A Web Application 
			to Create Local Climate Zone Maps. Front Environ Sci. 9. 
			https://doi.org/10.3389/fenvs.2021.637455


--------------------
DATA & FILE OVERVIEW
--------------------

Files and brief description:

	1. lcz_filter_v2.tif

	The recommended global LCZ map, compressed as GeoTIFF, with LCZ classes indicated by numbers 1-17. 
	LCZ labels are obtained after applying the morphological Gaussian filter described in Demuzere et al. (2020).
	The official color scheme - as indicated in the table below - is embedded into the GeoTIFF.
	
	The table below describes the class number | official class description | official hex color. 
	
	LCZ 1		|	Compact highrise	|	'#910613'	
	LCZ 2		|	Compact midrise		|	'#D9081C'
	LCZ 3		|	Compact lowrise		|	'#FF0A22'
	LCZ 4		|	Open highrise		|	'#C54F1E'
	LCZ 5		|	Open midrise		|	'#FF6628'
	LCZ 6		|	Open lowrise		|	'#FF985E'
	LCZ 7		|	Lightweight low-rise	|	'#FDED3F'
	LCZ 8		|	Large lowrise		|	'#BBBBBB'
	LCZ 9		|	Sparsely built		|	'#FFCBAB'
	LCZ 10		|	Heavy Industry		|	'#565656'
	LCZ 11 (A)	|	Dense trees		|	'#006A18'
	LCZ 12 (B)	|	Scattered trees		|	'#00A926'
	LCZ 13 (C)	|	Bush, scrub		|	'#628432'
	LCZ 14 (D)	|	Low plants		|	'#B5DA7F'
	LCZ 15 (E)	|	Bare rock or paved	|	'#000000'
	LCZ 16 (F)	|	Bare soil or sand	|	'#FCF7B1'
	LCZ 17 (G)	|	Water			|	'#656BFA'

	
	Other characteristics:
	- Projection: World Geodetic System 1984 (EPSG:4326)
	- Size: 389620, 155995
	- Spatial Resolution: 0.000898° (~ 100 m)
	- Representative for the nominal year of 2018


	2. lcz_v2.tif

	Same as 1., but presenting the raw LCZ map, before applying the morphological Gaussian filter. 


	3. lcz_probability_v2.tif

	A probability layer (%) that identifies how often the modal LCZ was chosen per pixel 
	(e.g. a probability of 60% means that the modal LCZ class was mapped 30 times out of 
	50 LCZ models). This is a pixel-based probability, derived from the lcz_v2.tif map.

--------------------
VERSIONS
--------------------

If there are there multiple versions of the dataset, list the file updated, when and why update was made:

	_v1: initial release of the data (20220222)
	_v2: add LCZ map of Iceland, consistent with the global LCZ map procedure (20221115)

