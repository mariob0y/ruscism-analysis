def get_message_text(msg):
    msg_type = msg.get("type")
    if msg_type != "message":
        return  # Ignore service messages
    if msg.get("forwarded_from"):
        return  # Ignore forwarded messages

    message_text = ""
    text_data = msg.get("text")

    if isinstance(text_data, str):
        message_text = text_data
    elif isinstance(text_data, list):
        for item in text_data:
            if isinstance(item, str):
                message_text += f" {item}"
            elif isinstance(item, dict) and item.get("type") not in [
                "link",
                "text_link",
            ]:
                message_text += f" {item.get('text', '')}"
    return message_text
