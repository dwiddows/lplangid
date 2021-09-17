# We might be surprised to find 'data' in a .py file, but it's small, well-curated using source control,
# and according to Turing and von Neumann, what we call "code" is just "the data that configures the machine".
#
# Each language configured here just has a set of words that should always be written as the top / most frequent
# words in that language, even if they're not very prevalent in (say) Wikipedia.

"""These words should go at the top of the ranking table for their language."""
TOP_DATA_OVERRIDES = {
    "en": ["bye", "hello", "hey", "hi", "I", "me", "no", "ok", "okay",
           "please", "pls", "thank", "thanks", "tks", "ty", "yes", "you"],
    "es": ["hola", "gracias", "no"],
    "it": ["no"],
    "ja": ["本"],
    "pt": ["obrigada", "obrigado"],
    "tl": ["po"],
    "zh": ["你好", "谢谢"]
}

"""These words should go in the stated position in their ranking tables - before the top data overrides are included."""
RANKED_DATA_OVERRIDES = {
    'de': {'bitte': 5, 'wir': 20, 'uns': 21, 'habe': 25, 'anliegen': 28, 'mir': 29, 'ihnen': 30, 'dir': 36, 'gerne': 38,
           'vielen': 41, 'mitarbeiter': 42, 'zufrieden': 52, 'mein': 63, 'bin': 65, 'guten': 67, 'antworten': 68,
           'weiter': 71, 'meine': 72, 'vertrag': 73, 'länger': 74, 'daher': 76, 'verbinden': 77, 'fragen': 79,
           'hast': 80, 'leider': 81, 'geht': 83, 'unseren': 84, 'melden': 88, 'nun': 92, 'unsere': 96, 'leite': 97},
    'en': {'chat': 61, 'payment': 76, 'customer': 77, 'conversation': 89, 'email': 95, 'wait': 100},
    'es': {'servicio': 19, 'consulta': 30, 'cuenta': 33, 'nuestro': 37, 'cliente': 39, 'información': 40, 'línea': 41,
           'banca': 44, 'estamos': 48, 'puedo': 49, 'móvil': 50, 'bac': 51, 'atención': 54, 'opción': 55, 'menú': 59,
           'momento': 62, 'horario': 63, 'puedes': 65, 'bienvenido': 66, 'solicitud': 69, 'buenas': 70, 'encuesta': 72,
           'nos': 77, 'conversación': 79, 'consultas': 80, 'escribe': 83, 'requerida': 85, 'servicios': 87,
           'nuestros': 88, 'minutos': 89, 'puntos': 90, 'ayudarte': 91, 'hacer': 92, 'envío': 93, 'responde': 94,
           'quiero': 95, 'días': 96, 'horarios': 98, 'saldo': 99, 'posible': 100},
    'fr': {'vous': 1, 'votre': 5, 'nous': 17, 'demande': 22, 'merci': 23, 'suis': 25, 'numéro': 34, 'bienvenue': 39,
           'conseiller': 53, 'nos': 55, 'notre': 57, 'lundi': 58, 'bonne': 60, 'oui': 63, 'conseillers': 65,
           'besoin': 68, 'journée': 70, 'terminer': 71, 'conversation': 72, 'répondent': 74, 'fermer': 76,
           'intéresse': 77, 'réponse': 78, 'satisfait': 79, 'transmise': 80, 'téléphone': 82, 'dossier': 85,
           'communiquer': 89, 'comment': 91, 'cela': 98},
    'it': {'visualizzabile': 9, 'grafico': 10, 'elemento': 11, 'ti': 16, 'utente': 17, 'tua': 21, 'numero': 25,
           'grazie': 30, 'mi': 31, 'ciao': 32, 'consulente': 33, 'tuo': 34, 'vuoi': 35, 'puoi': 38, 'richiesta': 40,
           'avere': 41, 'informazioni': 44, 'bisogno': 47, 'vodafone': 48, 'posso': 51, 'ci': 55, 'vorrei': 56,
           'servizio': 57, 'assistenza': 59, 'linea': 60, 'risponderà': 63, 'operatore': 65, 'contratto': 66,
           'supporto': 69, 'ricevere': 70, 'possibile': 71, 'esperto': 73, 'aiutarti': 74, 'gestire': 78, 'digita': 79,
           'nostro': 80, 'direttamente': 81, 'clienti': 83, 'passarti': 84, 'domanda': 87, 'notifica': 88,
           'risposta': 90, 'conto': 91, 'scrivi': 92, 'ringrazio': 95, 'contatto': 98, 'telefono': 99, 'attivo': 100},
    'nl': {'onze': 17, 'ons': 18, 'snel': 29, 'heb': 30, 'doen': 34, 'bedankt': 38, 'bericht': 44, 'huisnummer': 48,
           'postcode': 49, 'wij': 50, 'vragen': 54, 'langer': 59, 'storingen': 60, 'wachttijd': 61, 'ervaren': 63,
           'beantwoorden': 66, 'klantenservice': 67, 'verbind': 74, 'onderhoud': 76, 'sneller': 80, 'hebt': 82,
           'gegevens': 89, 'verwacht': 93, 'dankjewel': 94, 'hiervoor': 95, 'helpen': 97, 'collega': 98, 'nodig': 99},
    'pt': {'cartão': 15, 'atendimento': 19, 'digite': 22, 'fatura': 25, 'meu': 30, 'sim': 31, 'momento': 33,
           'seguro': 36, 'aqui': 37, 'opção': 39, 'voltar': 41, 'oi': 46, 'aguarde': 49, 'informações': 55,
           'informe': 57, 'falar': 58, 'sou': 59, 'posso': 62, 'assistente': 67, 'bom': 68, 'estou': 69, 'opções': 70,
           'desejada': 72, 'serviço': 75, 'quero': 79, 'limite': 80, 'preciso': 81, 'porto': 82, 'pedido': 89,
           'caso': 91, 'contato': 92, 'cadastro': 94, 'boa': 96, 'conta': 97, 'certo': 98, 'precisa': 99},
    'ru': {'вам': 4, 'помочь': 8, 'пожалуйста': 9, 'мы': 10, 'вы': 14, 'заказ': 16, 'меня': 17, 'здравствуйте': 18,
           'могу': 19, 'вас': 22, 'вопрос': 23, 'выберите': 24, 'вопросы': 26, 'мне': 28, 'чат': 29, 'зовут': 30,
           'привет': 32, 'спасибо': 33, 'списка': 38, 'день': 40, 'добрый': 41, 'поддержки': 43, 'звонке': 44,
           'водитель': 45, 'будет': 46, 'нет': 47, 'человека': 48, 'решить': 51, 'еще': 52, 'выбрать': 54, 'ниже': 56,
           'попробуйте': 57, 'связаться': 59, 'ваш': 60, 'помощь': 61, 'заказа': 62, 'готовы': 64, 'ваше': 65,
           'сейчас': 66, 'временно': 70, 'бот': 74, 'возможность': 75, 'нам': 78, 'сегодня': 80, 'служба': 81,
           'минут': 83, 'поездке': 84, 'обращение': 85, 'бота': 86, 'наши': 87, 'свои': 88, 'номер': 89, 'оставили': 90,
           'специалисты': 91, 'вами': 93, 'крутые': 94, 'приостановили': 95, 'специалистами': 96, 'ответами': 97,
           'задаваемые': 98, 'нажмите': 99, 'водителя': 100}}
