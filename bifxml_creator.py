import random

def create_bifxml_file(n):
    # Create an empty list for the variables
    variables = []

    # Generate n random variables with random probability tables
    for i in range(n):
        variables.append(f"v{i}")

    # Construct the .bifxml file
    bifxml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    bifxml += "<BIF version=\"0.3\">\n"
    bifxml += "\t<NETWORK>\n<NAME>example</NAME>\n"

    # Add the variables to the .bifxml file
    for variable in variables:
        bifxml += "<VARIABLE TYPE=\"nature\">\n"
        bifxml += f"<NAME>{variable}</NAME>\n"
        bifxml += "<OUTCOME>True</OUTCOME>\n"
        bifxml += "<OUTCOME>False</OUTCOME>\n"
        bifxml += "</VARIABLE>\n"

    for i, variable in enumerate(variables):
        bifxml += f"<DEFINITION>\n<FOR>{variable}</FOR>\n"

        if i>2:
            givens = random.sample(variables[:i], 2)
            for given in givens:       
                bifxml += f"<GIVEN>{given}</GIVEN>\n"
            p1 = random.random()
            p2 = random.random()
            p3 = random.random()
            p4 = random.random()
            bifxml += f"<TABLE>{p1} {1-p1} {p2} {1-p2} {p3} {1-p3} {p4} {1 -p4} </TABLE>\n</DEFINITION>\n"
        else:
            p1 = random.random()
            bifxml += f"<TABLE>{p1} {1-p1} </TABLE>\n</DEFINITION>\n"
    
    bifxml += "</NETWORK>\n</BIF>"

    # Open the file in write mode
    with open('myfile.bifxml', 'w') as f:
        f.write(bifxml)

def main():
    create_bifxml_file(1000)

if __name__ == "__main__":
    main()




