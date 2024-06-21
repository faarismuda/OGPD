import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def init_driver():
    options = Options()
    options.use_chromium = True
    driver_service = Service('path/to/msedgedriver')
    driver = webdriver.Edge(service=driver_service, options=options)
    return driver

def close_popup(driver):
    try:
        close_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".x92rtbv.x10l6tqk.x1tk7jg1.x1vjfegm"))
        )
        close_button.click()
        time.sleep(5)  # Wait a bit for the pop-up to close
    except Exception as e:
        print(f"Popup not found or already closed: {e}")

def click_see_more_buttons(driver):
    try:
        see_more_buttons = driver.find_elements(By.XPATH, "//div[text()='See more']")
        for button in see_more_buttons:
            driver.execute_script("arguments[0].click();", button)
            time.sleep(1)  # Wait a bit for the content to expand
    except Exception as e:
        print(f"No 'See more' buttons found or failed to click: {e}")

def scrape_posts(driver, username, limit):
    driver.get(f"https://web.facebook.com/{username}/")
    
    # Close authentication pop-up
    close_popup(driver)
    
    posts = []
    post_set = set()  # To keep track of unique posts
    css_selectors = [
        '.x1iorvi4.x1pi30zi.x1swvt13.xjkvuk6',
        '.x1iorvi4.x1pi30zi.x1l90r2v.x1swvt13',
        '.x1swvt13.x1pi30zi.xexx8yu.x18d9i69'
    ]
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while len(posts) < limit:
        click_see_more_buttons(driver)  # Click 'See more' buttons
        
        for selector in css_selectors:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = element.text.replace('\n', ' ')  # Replace newline characters with space
                if text not in post_set:
                    posts.append(text)
                    post_set.add(text)
                if len(posts) >= limit:
                    break
        
        if len(posts) >= limit:
            break
        
        # Scroll down to load more posts
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(5)  # Increased delay to 5 seconds to allow more time for posts to load
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    return posts[:limit]

def save_to_csv(posts, filename='facebook_posts.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Text'])
        for post in posts:
            writer.writerow([post])

def main(username, limit):
    driver = init_driver()
    try:
        posts = scrape_posts(driver, username, limit)
        save_to_csv(posts)
    finally:
        driver.quit()

if __name__ == "__main__":
    username = input("Enter the Facebook username: ")
    limit = int(input("Enter the number of posts to scrape: "))
    main(username, limit)
