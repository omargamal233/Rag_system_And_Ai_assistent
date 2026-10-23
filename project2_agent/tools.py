from langchain_core.tools import tool
from grades_data import GRADES
from typing import Dict

@tool
def add(a,b):
    """use when user ask to add two numbers."""
    if (isinstance(a, (int,float)) and isinstance(b, (int,float))):
        return a+b
    raise ValueError("numbers must be float or integer ")

@tool
def multiply(a,b):
    """use when user ask to multiply two numbers."""
    if (isinstance(a, (int,float)) and isinstance(b, (int,float))):
        return a*b
    raise ValueError("numbers must be float or integer ")
@tool
def div(a,b):
    """use when user ask to divide two numbers."""
    if(b ==0):
        raise ValueError("b cannot be zero")
    if (isinstance(a, (int,float)) and isinstance(b, (int,float))):
        return a/b
    raise ValueError("numbers must be float or integer ")
    
@tool
def sub(a,b):
    """use when user ask to subtract two numbers."""
    if (isinstance(a, (int,float)) and isinstance(b, (int,float))):
        return a-b
    raise ValueError("numbers must be float or integer ")
    
@tool
def time():
    """use when user ask for time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
@tool
def date():
    """use when user ask for date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

@tool
def get_student_grades(student_name: str)->Dict:
    """use when user ask for student grades."""
    if student_name in GRADES:
        return GRADES[student_name]    
    else:
        return f"Student {student_name} not found."