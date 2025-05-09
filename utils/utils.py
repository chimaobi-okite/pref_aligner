def get_messages(sys_message = None, user_message = None):
    if not user_message or user_message == "":
        print(user_message)
        print("Wrong user message format")
    messages = []
    if sys_message:
        messages.append({"role": "system", "content": sys_message})
    if user_message:
        # print("Entered user message if")
        messages.append({"role": "user", "content": user_message})
    return messages

def get_gemma_messages(sys_message = None, user_message = None):
    message = f"{sys_message}\n\n\n{user_message}"
    messages = []
    messages.append({"role": "user", "content": message})
    
    return messages