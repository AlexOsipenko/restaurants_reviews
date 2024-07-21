from openai import OpenAI


client = OpenAI(
    api_key = 'asdadaskdasdhjkashdkjdaks' #абракадабра, свой ключ, увы, предоставить не могу ¯\_(ツ)_/¯
)

def get_openai_response(prompt):
    response = client.chat.completions.create(
        model="davinci-002",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3,
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )
    return response.choices[0].message['content'].strip()

def main():
    print("Привет! Я Дмитрий Баринов из Кухни! Пока огузки делают свою стрепню - пойдем в курилку поболтаем: ")
    
    user_name = input("Как тебя зовут то? \n")
    print('Чего тебя ко мне привело то?\n*Баринов понимает только команды \"написать отзыв\", \"получить отзыв\". Чтобы закончить диалог - напиши \"пока\"*')
    
    while True:
        user_input = input(f"{user_name}: ")
        
        if user_input.lower() in ['пока']:
            print("Ну бывай! Заходи как нибудь на фирменную говядину в мятном маринаде с артишоками")
            break
        
        if "написать" in user_input.lower() and "отзыв" in user_input.lower():
            review_text = input("Дмитрий Баринов: хочешь рассказать мне о ресторане? Давай ка, давненько я из Клод Моне не выбирался...\nВведите название ресторана:")
            predicted_rating = get_openai_response(f"Оцени этот отзыв по шкале от 1 до 5: {review_text}")
            print(f"Дмитрий Баринов: Ну как я понял ты бы поставил этому заведению {predicted_rating.strip()} из 5? Понял, буду иметь ввиду. Чего еще обсудить хочешь?")

        elif "получить" in user_input.lower() and "отзыв" in user_input.lower():
            restaurant_name = input("Лучший ресторан это Клод Моне! Но есть и другие достойные заведения, о каком спросить хочешь?\nВведите название ресторана: ")
            review = get_openai_response(f"Покажи отзыв для ресторана {restaurant_name}")
            print(f"Дмитрий Баринов: Вот моё мнение для этой забегаловки: {review} О чем еще хочешь спросить?")
        else:
            print("Дмитрий Баринов: Не понимаю я тебя, дружище. Повтори вопрос.")

main()