"""Exercise the real video-to-video controls against an isolated local server."""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:47823')
    parser.add_argument('--screenshots', type=Path, required=True)
    parser.add_argument('--sample-video', type=Path)
    parser.add_argument('--sample-image', type=Path)
    parser.add_argument('--gallery-qa', action='store_true')
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 900}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url.rstrip('/') + '/#reference', wait_until='networkidle')
        page.get_by_role('button', name='Video to video').click()
        page.get_by_role('heading', name='Video to video').wait_for()
        assert page.get_by_role('tab', name='Replace subjects').get_attribute('aria-selected') == 'true'
        assert not page.get_by_text('Camera position', exact=True).is_visible()
        page.screenshot(path=str(args.screenshots / 'replace-subjects.png'), full_page=True)

        page.get_by_role('tab', name='Camera view').click()
        assert page.get_by_text('Camera position', exact=True).is_visible()
        disclosure = page.locator('details').filter(has=page.locator('summary', has_text='Replace subjects before changing camera view'))
        assert disclosure.count() == 1
        assert not disclosure.evaluate('(element) => element.open')
        assert not page.get_by_label('Replace subjects first').is_visible()
        disclosure.locator('summary').click()
        page.get_by_label('Replace subjects first').check()
        assert page.get_by_label('Subject replacement prompt').is_visible()
        assert page.get_by_role('button', name='Add image').is_visible()
        page.screenshot(path=str(args.screenshots / 'camera-with-optional-swap.png'), full_page=True)
        page.get_by_label('Replace subjects first').uncheck()
        assert not page.get_by_label('Subject replacement prompt').is_visible()
        disclosure.locator('summary').click()
        assert not page.get_by_label('Replace subjects first').is_visible()
        page.get_by_role('tab', name='Replace subjects').click()
        assert not page.get_by_text('Camera position', exact=True).is_visible()

        page.get_by_role('button', name='Upscale', exact=True).click()
        assert not page.get_by_label('Keep source audio').is_visible()
        page.get_by_text('Advanced settings', exact=False).last.click()
        assert page.get_by_label('Keep source audio').is_visible()
        page.screenshot(path=str(args.screenshots / 'upscale.png'), full_page=True)
        assert not errors, errors
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), page.evaluate("[...document.querySelectorAll('*')].filter(e => e.getBoundingClientRect().right > innerWidth + 1).slice(0,12).map(e => [e.tagName, e.className, Math.round(e.getBoundingClientRect().right)])")

        page.set_viewport_size({'width': 768, 'height': 850})
        page.get_by_role('button', name='Video to video').click()
        page.get_by_role('tab', name='Camera view').click()
        page.screenshot(path=str(args.screenshots / 'camera-tablet.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), page.evaluate("[...document.querySelectorAll('*')].filter(e => e.getBoundingClientRect().right > innerWidth + 1).slice(0,12).map(e => [e.tagName, e.className, Math.round(e.getBoundingClientRect().right)])")

        if args.sample_video and args.sample_image:
            page.set_viewport_size({'width': 1440, 'height': 900})
            page.get_by_role('tab', name='Replace subjects').click()
            page.locator('main.reference-page input[type=file][accept^="video"]').first.set_input_files(str(args.sample_video))
            page.get_by_text(args.sample_video.name, exact=True).wait_for(timeout=20000)
            page.locator('main.reference-page input[type=file][accept^="image"]').first.set_input_files(str(args.sample_image))
            page.get_by_alt_text('Reference 1').wait_for(timeout=20000)
            page.get_by_role('tab', name='Camera view').click()
            disclosure = page.locator('details').filter(has=page.locator('summary', has_text='Replace subjects before changing camera view'))
            disclosure.locator('summary').click()
            page.get_by_label('Replace subjects first').check()
            assert page.get_by_alt_text('Reference 1').is_visible()
            page.screenshot(path=str(args.screenshots / 'camera-with-inputs.png'), full_page=True)

        if args.gallery_qa:
            page.get_by_role('button', name='Gallery', exact=True).click()
            page.locator('.gallery-item').first.wait_for()
            assert page.locator('.gallery-item').count() >= 10
            assert page.locator('.gallery-source-indicator').count() >= 10
            page.wait_for_function("[...document.querySelectorAll('.gallery-preview > img')].every(image => image.complete && image.naturalWidth > 0)")
            page.screenshot(path=str(args.screenshots / 'gallery-ten-results.png'), full_page=True)
            page.locator('.gallery-item .gallery-preview').first.click()
            assert page.get_by_role('dialog', name='Generation details').is_visible()
            assert page.locator('.gallery-detail .gallery-source-video').is_visible()
            assert page.get_by_role('button', name='Use settings').last.is_visible()
            page.get_by_role('button', name='Close generation details').click()
            assert page.locator('.gallery-item').count() >= 10
            page.locator('.gallery-item .gallery-preview').nth(1).click()
            intermediate = page.locator('.gallery-detail .gallery-intermediate')
            assert intermediate.is_visible()
            intermediate.locator('summary').click()
            assert intermediate.locator('video').is_visible()
            video_url = intermediate.locator('video').get_attribute('src')
            assert page.request.get(args.url.rstrip('/') + video_url).status == 200
        browser.close()
    print('UI mode separation, disclosure controls, desktop/tablet layout, and browser errors: passed')


if __name__ == '__main__':
    main()
