The five tasks are written in the same document assignment.py where each function is called by flags.
Open the file with a code editor that supports python such as PyCharm, Visual Studio, or the terminal.
As long as the file can be read, no IDE is required. 

#Task 1

This file contains the instructions for runnig tasks 1 to 5.
The instructions to run the Task 1 program are detailed below

Open the file named 'the file name' with a Python lDE such as PyCharm or Visual Studio code.

The 4 graphs are printed out by typing their corresponding flag names in the terminal of the lDE.

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


Task 2

To run the 'Defaunt' model, type -defaunt in the terminal which will call the 'defaunt_main' function
The main function has two inputs 'threshold' and 'beta', which are set to 0.2 as default values.
The values can be altered with flags '-threshold' and 'beta' followed by a space and the preferred value.
For testing the model, use the flag 'test_defaunt' to call the 'defaunt_test' function and print out changes made to each opinion.

It should be noted that the graphs may take over a minute to generate when population exceeds 50 as it is a random integer with an upper bound of 100.

#Task 3

The instruction to run the task3 program:

Open the file name ‘the name’ with a Python IDE such as PyCharm or Visual Studio code

The network graph and the data of mean degree, mean path length and clustering coefficient is printed by their corresponding flag names in the terminal of the IDE.

In the terminal, type ‘python3 FCP_task_3.py-’followed by the name at the end.

For example, ‘python3 FCP_task_3.py-network 10’(Once you print the‘python3 FCP_task_3.py-network 10’ you will get the graph and the the data of mean degree, mean path length and clustering coefficient)

The names of two flags are:

• network 10  
• test_network      

Note: If you could not see  the graph on the window, try to move the mouse to the window, the graph will be shown.


#Task 4

The instructions to run the Task 4 program are detailed below
Open the file named ‘the name’ with a Python IDE such as PyCharm or Visual Studio code. 
The 3 network graphs are printed out by typing their corresponding flag names in the terminal of the IDE.
 In the terminal, type “python3 FCP_task_4.py -” followed by the flag name at the end.
For example, “python3 FCP_task_4.py -ring_network 10”
The names of the three flags are:
•	ring_network 10
•	small_world 10
•	small_world 10 -re_wire 0.1

Note that the value followed by re_wire is variable and ranges between 0.2 and 0.98.

Each network graph is going to be opened in separate windows.
 — “python3 FCP_task_4.py -ring_network 10”, “python3 FCP_task_4.py -small_world 10” and “python3 FCP_task_4.py -small_world 10 -re_wire 0.1”
 
Finally for selecting neighbours from a network with the Defuant model, type in -defaunt -use_network
The Defaunt model would still select lists as inputs by default, but would also accept networks and nodes.
