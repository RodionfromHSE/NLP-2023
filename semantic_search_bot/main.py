import logging
import json
from collections import deque, defaultdict
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from semantic_searcher import SemanticSearcher

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

ss = SemanticSearcher()

# A dictionary to store the last 100 messages for each chat
chat_histories = defaultdict(lambda: deque(maxlen=100))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Welcome! This bot can find semantically similar messages. Use /find <your_query>.')

async def find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /find command."""
    chat_id = update.effective_chat.id
    user_query = update.message.text.partition(' ')[2] # Extract query

    if not user_query:
        await update.message.reply_text("Please provide a query after the /find command.")
        return

    if chat_id not in chat_histories or len(chat_histories[chat_id]) == 0:
        await update.message.reply_text("No recent messages found in the chat.")
        return

    # last 100 messages
    messages = list(chat_histories[chat_id])
    similar_messages, scores = ss(user_query, messages, return_scores=True)
    def to_message(score, message):
        begin = f"Message with score {score:.2f}:\n"
        end = f"\n{'-'*20}\n"
        return begin + message + end
    top_3 = [to_message(score, message) for score, message in zip(scores, similar_messages)][:3]
    response = '\n'.join(top_3)
    await update.message.reply_text(response)

async def store_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Store the user message."""
    chat_id = update.effective_chat.id
    message_text = update.message.text
    chat_histories[chat_id].append(message_text)

def main() -> None:
    token = json.load(open('token.json'))['token']
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("find", find))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, store_message))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == '__main__':
    main()
