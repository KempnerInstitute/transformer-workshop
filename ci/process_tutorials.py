# This is adapted from Neuromatch Academy's CI scripts (BSD 3-Clause License) 
# https://github.com/NeuromatchAcademy/nmaci/blob/main/scripts/process_notebooks.py
 
import nbformat
import os
import hashlib
import sys
from copy import deepcopy

GITHUB_URL = (
    f"https://github.com/KempnerInstitute/transformer-workshop/blob/main/"
)

COLAB_URL = (
    f"https://colab.research.google.com/github/KempnerInstitute/transformer-workshop/blob/main/"
)

def main():

    # Find base tutorial notebook files
    nb_paths = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if "_base" in file and file.endswith(".ipynb"):
                nb_paths.append(os.path.join(root, file))
    
    
    for nb_path in nb_paths:

        # Load the notebook structure
        with open(nb_path) as f:
            nb = nbformat.read(f, nbformat.NO_CONVERT)

        # Extract components of the notebook path
        nb_dir, nb_fname = os.path.split(nb_path)
        nb_name, _ = os.path.splitext(nb_fname)
        nb_name = nb_name.split('_base')[0]

        # Create subdirectories, if they don't exist
        solutions_dir = make_sub_dir(nb_dir, "solutions")
        hints_dir = make_sub_dir(nb_dir, "hints")

        # Update or add colab badge and save
        add_colab_badge(nb, nb_fname)
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)


        # Remove outputs from notebook before saving student and instructor version
        nb_cells = nb.get("cells", [])
        for i, cell in enumerate(nb_cells):
            if 'outputs' in cell.keys():        
                cell['outputs'] = []

        # Generate the student version and save it to a subdirectory
        print(f"Extracting solutions from {nb_path}")
        student_nb, solution_snippets, hint_snippets = extract_solutions_and_hints(nb, nb_dir, nb_name)

        # Generate the instructor version 
        print(f"Create instructor notebook from {nb_path}")
        instructor_nb = remove_code_exercises(nb, nb_dir, nb_name)

        # Add (if necessary) and update colab badges in each notebook
        add_colab_badge(student_nb, f"{nb_name.split('.')[0]}_student.ipynb")
        add_colab_badge(instructor_nb, f"{nb_name.split('.')[0]}_instructor.ipynb")

        # Save student notebook
        student_nb_path = f"{nb_name.split('.')[0]}_student.ipynb"
        print(f"Writing student notebook to {student_nb_path}")
        with open(student_nb_path, "w") as f:
            nbformat.write(student_nb, f)

        # Save instructor notebook
        instructor_nb_path = f"{nb_name.split('.')[0]}_instructor.ipynb"
        print(f"Writing instructor notebook to {instructor_nb_path}")
        with open(instructor_nb_path, "w") as f:
            nbformat.write(instructor_nb, f)

        # Write the solution snippets
        print(f"Writing solution snippets to {solutions_dir}")
        for fname, snippet in solution_snippets.items():
            fname = fname.replace("solutions", solutions_dir)
            with open(fname, "w") as f:
                f.write(snippet)

        # Write the hint snippets
        print(f"Writing hint snippets to {hints_dir}")
        for fname, snippet in hint_snippets.items():
            fname = fname.replace("solutions", hints_dir)
            with open(fname, "w") as f:
                f.write(snippet)


def make_sub_dir(nb_dir, name):
    """Create nb_dir/name if it does not exist."""
    sub_dir = os.path.join(nb_dir, name)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    return sub_dir

def extract_solutions_and_hints(nb, nb_dir, nb_name):
    nb = deepcopy(nb)

    solution_snippets = {}
    hint_snippets = {}

    nb_cells = nb.get("cells", [])
    for i, cell in enumerate(nb_cells):
        cell_text = cell["source"].replace(" ", "").lower()

        is_hint_cell = cell_text.startswith("#hint") or cell_text.startswith("##hint") or cell_text.startswith("Hint") or cell_text.startswith("hint") or cell_text.startswith("**hint")
        is_solution_cell = cell_text.startswith("#solution") or cell_text.startswith("##solution")  or cell_text.startswith("Solution")
        if is_hint_cell or is_solution_cell:

            # Get the cell source
            cell_source = cell["source"]

            # Hash the source to get a unique identifier
            cell_id = hashlib.sha1(cell_source.encode("utf-8")).hexdigest()[:8]

            # Clean up the cell source and assign a filename
            if is_solution_cell:
                snippet = "\n".join(cell_source.split("\n")[1:])
                py_fname = f"solutions/{nb_name}_Solution_{cell_id}.py"
                solution_snippets[py_fname] = snippet

                # Convert the solution cell to markdown,
                # Insert a link to the solution snippet script on github
                py_url = f"{GITHUB_URL}/{py_fname}"
                new_source = f"[*Click for solution*]({py_url})\n\n"

            elif is_hint_cell:
                snippet = "\n".join(cell_source.split("\n")[1:])
                py_fname = f"hints/{nb_name}_Hint_{cell_id}.md"
                hint_snippets[py_fname] = snippet

                # Convert the solution cell to markdown,
                # Insert a link to the solution snippet script on github
                py_url = f"{GITHUB_URL}/{py_fname}"
                hint_text = cell_source.split("\n")[0].replace('*', '').lower()
                new_source = f"[*Click for Reflect and Discuss solutions*]({py_url})\n\n"


            cell["source"] = new_source
            cell["cell_type"] = "markdown"
            cell["metadata"]["colab_type"] = "text"
            if "outputID" in cell["metadata"]:
                del cell["metadata"]["outputId"]
            if "outputs" in cell:
                del cell["outputs"]
            if "execution_count" in cell:
                del cell["execution_count"]

    return nb, solution_snippets, hint_snippets


def remove_code_exercises(nb, nb_dir, nb_name):
    """Convert notebook to instructor notebook."""
    nb = deepcopy(nb)
    _, tutorial_dir = os.path.split(nb_dir)

    nb_cells = nb.get("cells", [])
    for i, cell in enumerate(nb_cells):

        cell_text = cell["source"].replace(" ", "").lower()
        has_code_exercise = cell_text.startswith("#solution") or cell_text.startswith("##solution")
        if has_code_exercise:
            is_markdown = True
            cell_id = i
            while is_markdown:
                
                is_markdown = nb_cells[cell_id-1]["cell_type"] == "markdown"
                cell_id -= 1

            nb_cells[cell_id]["cell_type"] = "markdown"
            nb_cells[cell_id]["metadata"]["colab_type"] = "text"
            if "outputID" in nb_cells[cell_id]["metadata"]:
                del nb_cells[cell_id]["metadata"]["outputId"]
            if "outputs" in nb_cells[cell_id]:
                del nb_cells[cell_id]["outputs"]
            if "execution_count" in nb_cells[cell_id]:
                del nb_cells[cell_id]["execution_count"]

            nb_cells[cell_id]['source'] = '```python\n' + nb_cells[cell_id]['source']+'\n\n```'

    return nb


def add_colab_badge(nb, fname):

    nb_cells = nb.get("cells", [])
    has_colab_badge = "colab-badge.svg" in nb_cells[0]['source']

    correct_url = f"{COLAB_URL}{fname}"
    source_txt = f'<a href="{correct_url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'

    if has_colab_badge:
        nb.cells[0]['source'] = source_txt
    else:
        badge_cell = nbformat.v4.new_markdown_cell(source=source_txt)
        badge_cell["metadata"] = {"id": "view-in-github", "colab_type": "text"}

        nb.cells.insert(0, badge_cell)



if __name__ == "__main__":

    main()
