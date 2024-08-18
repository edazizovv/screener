
#
import os
import time

#
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.common.exceptions import InvalidSessionIdException
from selenium.webdriver.firefox.service import Service


try:
    default_save_dir = 'C:/Users/Edward/Downloads/'
    target_save_dir = './data/futures/raw/'

    executable_path = 'C:\\TET\\geckodriver-v0.31.0-win64\\geckodriver.exe'
    # s = Service(executable_path)
    s = Service()
    options = Options()
    options.add_argument("--headless")
    # options.add_argument("--window-size=1920,1080")
    # options.add_argument('--ignore-certificate-errors')
    # options.add_argument('--allow-running-insecure-content')

    driver = webdriver.Firefox(service=s, options=options)

    query_url = 'https://data.binance.vision/?prefix=data/futures/um/daily/klines/'
    driver.get(query_url)

    time.sleep(5)

    nein = "//*[contains(text(), 'BTCUSDT')]"
    elems = driver.find_elements(By.XPATH, nein)
    n = len(elems)
    print("Found {0} futures, going through".format(n))

    for j in range(n):
        if j not in []:
            nein = "//*[contains(text(), 'BTCUSDT')]"
            elems = driver.find_elements(By.XPATH, nein)
            elem = elems[j]
            _name = elem.get_attribute('text')
            print("\tChecking {0}".format(_name))
            # print("\t\tWaiting")
            # print("\t\tClicked")
            driver.execute_script("arguments[0].scrollIntoView();", elem)
            time.sleep(1)
            elem.click()

            nein = "//*[contains(text(), '1h')]"
            WebDriverWait(driver, 1000000).until(
                expected_conditions.presence_of_element_located((By.XPATH, nein)))

            gelem = driver.find_element(By.XPATH, nein)
            WebDriverWait(driver, 1000000).until(
                expected_conditions.element_to_be_clickable(gelem)).click()

            nein = "//*[contains(text(), '{0}-1h') and not(contains(text(), 'CHECKSUM'))]".format(_name[:_name.index('/')])
            WebDriverWait(driver, 1000000).until(
                expected_conditions.presence_of_element_located((By.XPATH, nein)))

            velems = driver.find_elements(By.XPATH, nein)

            # print("\t\tFound {0} subitems".format(len(velems)))
            for velem in velems:
                driver.execute_script("arguments[0].scrollIntoView();", velem)
                WebDriverWait(driver, 1000000).until(
                    expected_conditions.element_to_be_clickable(velem)).click()
                time.sleep(0.5)
                name = velem.get_attribute('text')
                os.replace(default_save_dir + name, target_save_dir + name)
                print('\t\tSaving to {0}'.format(target_save_dir + name))

            driver.back()
            driver.back()
            print("\t\t{0} is done".format(_name))
            time.sleep(5)

except Exception as e:
    raise e
