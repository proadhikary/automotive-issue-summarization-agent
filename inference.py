import api

while True:
    user_input = input("Paste Json here: ")
    if user_input == "exit":
        break
    response = api.generate_response(user_input)
    print(response)

