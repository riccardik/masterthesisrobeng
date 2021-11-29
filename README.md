# masterthesisrobeng

## Collect the datasets
Run the simulation

	C:\Users\rick\Desktop\SOFA_v20.12.00_Win64\bin\runSofa.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\vein_deformation_kidney.pyscn"

run the position control script for the kidney
	
	c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\send_position_rotation.py"


	
prepare the datasets
parse the data

	c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\parse_data.py"
	
	c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\separate_bycol.py"
	
build the final normalized datasets:	

	c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\tensorflow\tutorial\normalizeallsets.py"

## Create the model
	regression regressionnonnorm/nonnorm2/nonnormart
	rnn rnnart
	
## inference the test sets
	lstmvein_final  lstmart_final
	
	nnart_final nnvein_final
	
watch the animation
			c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\3dplotspline.py"

	

## inference from the model in real time
Run the simulation

	C:\Users\rick\Desktop\SOFA_v20.12.00_Win64\bin\runSofa.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\vein_deformation_kidney.pyscn"

run the position control script for the kidney
	
	c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\send_position_rotation.py"

run the script to inference in real time
	
	c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\receive_data_spline.py"