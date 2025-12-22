import streamlit as st

def main():
    st.title("Streamlit App")
    
    # Display some text
    st.write("Hello, Streamlit!")
    
    # Input field
    user_input = st.text_input("Enter something")
    
    # Display the user input
    if user_input:
        st.write(f"You entered: {user_input}")

if __name__ == "__main__":
    main()