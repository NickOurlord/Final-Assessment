This file contains the instructions for runnig tasks 1 to 5.
The instructions to run the Task 1 program are detailed below

The 4 graphs for task1 are printed out by typing their corresponding flag names in the terminal of the lDE.

In the terminal, type “python3 FCP_assignment_Task_1.py -" followed by the flag name at the end.

For example, “python3 FCP_assignment_Task_1.py -ising_model”
The names of the 4 flags are:
	-ising_model
	-ising_model -external -0.1
	-ising_model -alpha 10
	-test_ising


Note that the value followed by external and alpha are variable. The code should default to using a value of H = 0 and a value of alpha = 1, unless you also include the flag -external -<H> and -alpha <alpha>.

Each graph is going to be opened in separate windows.
"python3 FCP_assignment_Task_1.py -ising_model", “python3 FCP_assignment_Task_1.py -ising_model -external -0.1", “python3 FCP_assignment_Task_1.py -ising_model -alpha 10”, and “python3 FCP_assignment_Task_1.py -test_ising”.



The five tasks are written in the same document assignment.py where each function is called by flags.
Open the file with a code editor that supports python such as PyCharm, Visual Studio, or the terminal.
As long as the file can be read, no IDE is required. 

To run the 'Defaunt' model, type -defaunt in the terminal which will call the 'defaunt_main' function
The main function has two inputs 'threshold' and 'beta', which are set to 0.2 as default values.
The values can be altered with flags '-threshold' and 'beta' followed by a space and the preferred value.
For testing the model, use the flag 'test_defaunt' to call the 'defaunt_test' function and print out changes made to each opinion.

It should be noted that the graphs may take over a minute to generate when population exceeds 50 as it is a random integer with an upper bound of 100.





