def get_messages(sys_message, user_message):
    messages = []
    if sys_message:
        messages.append({"role": "system", "content": sys_message})
    
    messages.append({"role": "user", "content": user_message})
    return messages