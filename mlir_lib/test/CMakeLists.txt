configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(PADDLE_OPT_TEST_DEPENDS FileCheck count not paddle-opt)

add_lit_testsuite(
  check-paddle-opt "Running the paddle-opt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${PADDLE_OPT_TEST_DEPENDS})
set_target_properties(check-paddle-opt PROPERTIES FOLDER "Tests")

# TODO restore it
# add_lit_testsuites(PADDLE_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
#                    ${PADDLE_OPT_TEST_DEPENDS})
