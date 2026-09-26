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
    parser.add_argument('--two-person-video', type=Path)
    parser.add_argument('--gallery-qa', action='store_true')
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=str(Path.home() / '.cloakbrowser/chromium-146.0.7680.177.5/chrome.exe'), args=['--disable-gpu'])
        page = browser.new_page(viewport={'width': 1440, 'height': 900}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url.rstrip('/') + '/#video-to-video', wait_until='networkidle')
        assert page.locator('.app-tabs button').count() == 3
        assert page.locator('.app-tabs').get_by_role('button', name='Video', exact=True).is_visible()
        assert page.get_by_role('tab', name='Video to video').get_attribute('aria-selected') == 'true'
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
        assert not page.get_by_label('Additional replacement instructions (optional)').is_visible()
        assert page.get_by_role('button', name='Add image').is_visible()
        page.screenshot(path=str(args.screenshots / 'camera-with-optional-swap.png'), full_page=True)
        page.get_by_label('Replace subjects first').uncheck()
        assert not page.get_by_label('Additional replacement instructions (optional)').is_visible()
        disclosure.locator('summary').click()
        assert not page.get_by_label('Replace subjects first').is_visible()
        page.get_by_role('tab', name='Replace subjects').click()
        assert not page.get_by_text('Camera position', exact=True).is_visible()

        page.get_by_role('tab', name='Create video').click()
        assert page.get_by_role('button', name='Text to video', exact=True).is_visible()
        assert not page.get_by_label('Include generated audio').is_visible()
        page.locator('.video-advanced summary').click()
        assert page.get_by_label('Include generated audio').is_visible()

        page.get_by_role('tab', name='Upscale', exact=True).click()
        assert not page.get_by_label('Keep source audio').is_visible()
        page.get_by_text('Advanced settings', exact=False).last.click()
        assert page.get_by_label('Keep source audio').is_visible()
        page.screenshot(path=str(args.screenshots / 'upscale.png'), full_page=True)
        assert not errors, errors
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), page.evaluate("[...document.querySelectorAll('*')].filter(e => e.getBoundingClientRect().right > innerWidth + 1).slice(0,12).map(e => [e.tagName, e.className, Math.round(e.getBoundingClientRect().right)])")

        page.set_viewport_size({'width': 768, 'height': 850})
        page.get_by_role('tab', name='Video to video').click()
        page.get_by_role('tab', name='Camera view').click()
        page.screenshot(path=str(args.screenshots / 'camera-tablet.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), page.evaluate("[...document.querySelectorAll('*')].filter(e => e.getBoundingClientRect().right > innerWidth + 1).slice(0,12).map(e => [e.tagName, e.className, Math.round(e.getBoundingClientRect().right)])")

        if args.sample_video and args.sample_image:
            page.set_viewport_size({'width': 1440, 'height': 900})
            page.get_by_role('tab', name='Replace subjects').click()
            page.locator('main.reference-page input[type=file][accept^="video"]').first.set_input_files(str(args.sample_video))
            page.get_by_text(args.sample_video.name, exact=True).wait_for(timeout=20000)
            aspect = page.locator('#reference-aspect')
            tier = page.locator('#reference-size-tier')
            assert aspect.input_value() == 'auto' and tier.input_value() == 'standard'
            def output_size():
                return [int(part.strip()) for part in page.locator('.reference-resolution-result strong').inner_text().split('×')]
            width, height = output_size()
            assert abs(width / height - 9/16) < 0.02, (width, height)
            aspect.select_option('16:9')
            width, height = output_size()
            assert abs(width / height - 16/9) < 0.03, (width, height)
            tier.select_option('draft')
            draft_area = output_size()[0] * output_size()[1]
            tier.select_option('high')
            high_area = output_size()[0] * output_size()[1]
            assert high_area > draft_area
            aspect.select_option('custom')
            page.get_by_label('Ratio width').fill('5')
            page.get_by_label('Ratio height').fill('4')
            page.get_by_label('Ratio height').press('Tab')
            width, height = output_size()
            assert abs(width / height - 1.25) < 0.03, (width, height)
            tier.select_option('custom')
            page.locator('.reference-resolution-custom').get_by_label('Width', exact=True).fill('640')
            page.locator('.reference-resolution-custom').get_by_label('Height', exact=True).fill('640')
            assert output_size() == [640, 640]
            aspect.select_option('auto')
            tier.select_option('standard')
            width, height = output_size()
            assert abs(width / height - 9/16) < 0.02, (width, height)
            page.locator('#reference-model').select_option('h3-ref-eros-q4')
            assert output_size() == [width, height]
            page.locator('#reference-model').select_option('h3-ref-turbo-int8')
            page.get_by_role('slider', name='Trim start').wait_for(timeout=20000)
            assert page.get_by_role('button', name='Play selection').is_visible()
            assert page.get_by_role('heading', name='Source preview').is_visible()
            assert page.locator('.reference-output video').is_visible()
            page.wait_for_function("document.querySelector('.clip-strip').complete && document.querySelector('.clip-strip').naturalWidth > 0", timeout=35000)
            assert page.locator('.clip-strip').get_attribute('src').endswith('/timeline')
            original_end = float(page.get_by_label('Clip end in seconds').input_value())
            assert original_end > 10, original_end
            page.get_by_label('Clip start in seconds').fill('2.00')
            page.get_by_label('Clip start in seconds').press('Enter')
            page.wait_for_function("Math.abs(parseFloat(document.querySelector('[aria-label=\"Clip start in seconds\"]').value)-2)<0.05")
            page.get_by_label('Clip end in seconds').fill('6.00')
            page.get_by_label('Clip end in seconds').press('Enter')
            page.wait_for_function("Math.abs(parseFloat(document.querySelector('[aria-label=\"Clip end in seconds\"]').value)-6)<0.05")
            page.get_by_role('button', name='Play selection').click()
            page.wait_for_function("document.querySelector('.reference-source-editor video').currentTime >= 1.95")
            with page.expect_response(lambda response: '/people' in response.url and response.request.method == 'POST', timeout=30000):
                page.get_by_role('button', name='Scan selected clip').click()
            page.get_by_text('Finding people across the selected clip…').wait_for(state='hidden', timeout=30000)
            assert page.get_by_text('@Person1', exact=True).is_visible()
            page.locator('main.reference-page input[type=file][accept^="image"]').first.set_input_files(str(args.sample_image))
            page.get_by_alt_text('Reference 1').first.wait_for(timeout=20000)
            assert page.get_by_text('@Ref1', exact=True).first.is_visible()
            assert page.get_by_role('button',name='Replacement for Person 1').get_by_alt_text('Selected reference 1').is_visible()
            prompt = page.get_by_label('Additional instructions (optional)', exact=True)
            page.locator('.reference-inputs > .reference-extra-instructions > summary').click()
            prompt.fill('Replace @')
            page.get_by_role('option', name='@Person1 Person 1').click()
            prompt.press('End')
            prompt.type('with @')
            page.get_by_role('option', name='@Ref1 Reference 1').click()
            assert '@Person1' in prompt.input_value() and '@Ref1' in prompt.input_value()
            page.screenshot(path=str(args.screenshots / 'clip-and-person-mapping.png'), full_page=True)
            page.get_by_role('tab', name='Camera view').click()
            disclosure = page.locator('details').filter(has=page.locator('summary', has_text='Replace subjects before changing camera view'))
            disclosure.locator('summary').click()
            page.get_by_label('Replace subjects first').check()
            assert page.get_by_alt_text('Reference 1').first.is_visible()
            replacement_prompt = page.get_by_label('Additional replacement instructions (optional)')
            disclosure.locator('.reference-extra-instructions > summary').click()
            replacement_prompt.fill('Replace @')
            page.get_by_role('option', name='@Person1 Person 1').click()
            replacement_prompt.press('End')
            replacement_prompt.type('with @')
            page.get_by_role('option', name='@Ref1 Reference 1').click()
            assert '@Person1' in replacement_prompt.input_value() and '@Ref1' in replacement_prompt.input_value()
            page.screenshot(path=str(args.screenshots / 'camera-with-inputs.png'), full_page=True)
            if args.two_person_video:
                page.get_by_role('tab', name='Replace subjects').click()
                with page.expect_response(lambda response: '/people' in response.url and response.request.method == 'POST', timeout=30000):
                    page.locator('main.reference-page input[type=file][accept^="video"]').first.set_input_files(str(args.two_person_video))
                page.get_by_label('Replacement for Person 2').wait_for(timeout=30000)
                page.get_by_role('button',name='Replacement for Person 2').click()
                page.get_by_role('option',name='@Ref1',exact=True).click()
                assert page.get_by_role('button',name='Replacement for Person 1').inner_text().strip() == 'Keep person'
                page.get_by_role('button', name='Play selection').click()
                page.wait_for_function("document.querySelector('.reference-source-editor video').currentTime > 0.1", timeout=15000)
                page.screenshot(path=str(args.screenshots / 'two-person-mapping.png'), full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(args.screenshots / 'replace-mobile-with-inputs.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.set_viewport_size({'width':1440,'height':900})

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
            page.locator('.gallery-detail-actions').get_by_role('button', name='Use settings').click()
            assert page.get_by_role('tab', name='Video to video').get_attribute('aria-selected') == 'true'
            assert page.locator('.app-tabs').get_by_role('button', name='Video', exact=True).get_attribute('aria-current') == 'page'
        browser.close()
    print('UI mode separation, disclosure controls, desktop/tablet layout, and browser errors: passed')


if __name__ == '__main__':
    main()
