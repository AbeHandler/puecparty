describe('Instructor Search Functionality', () => {

  beforeEach(() => {
    // Visit the application before each test
    cy.visit('/')
  })

  it('should successfully search for instructor Cunningham, Cory Brooke from Fall 2022 to Spring 2025', () => {
    // Fill in the instructor name
    cy.get('#instructor').type('Cunningham, Cory Brooke')

    // Select start year (Fall 2022)
    cy.get('#startYear').select('2022')

    // The end year should auto-populate based on PUEC mode (default)
    // Verify it shows Spring 2025 (PUEC mode: start year + 3)
    cy.get('#endYearDisplay').should('contain', 'Spring 2025')

    // Submit the form
    cy.get('#submitBtn').click()

    // Wait for results to load (check for loading spinner to disappear)
    cy.get('.spinner-border', { timeout: 15000 }).should('not.exist')

    // Verify results are displayed
    cy.get('#results').should('be.visible')

    // Verify no error messages
    cy.get('.alert-danger').should('not.exist')

    // Verify success indicators are present
    cy.get('.card-header.bg-success').should('exist')

    // Verify that results contain expected elements
    cy.get('.accordion').should('exist')
  })

  it('should handle the search with Ad hoc mode for custom date range', () => {
    // Switch to Ad hoc mode
    cy.get('#adhocBtn').click()

    // Fill in the instructor name
    cy.get('#instructor').type('Cunningham, Cory Brooke')

    // Select start year (Fall 2022)
    cy.get('#startYear').select('2022')

    // Select end year (Spring 2025) - visible in Ad hoc mode
    cy.get('#endYear').select('2025')

    // Submit the form
    cy.get('#submitBtn').click()

    // Wait for results to load
    cy.get('.spinner-border', { timeout: 15000 }).should('not.exist')

    // Verify results are displayed
    cy.get('#results').should('be.visible')

    // Verify no error messages
    cy.get('.alert-danger').should('not.exist')

    // Verify success indicators
    cy.get('.card-header.bg-success').should('exist')
  })

  it('should display an error when no instructor is selected', () => {
    // Don't fill in instructor name

    // Select start year
    cy.get('#startYear').select('2022')

    // Submit the form
    cy.get('#submitBtn').click()

    // Wait for error message
    cy.get('.alert-danger', { timeout: 10000 }).should('be.visible')
    cy.get('.alert-danger').should('contain', 'Please enter an instructor name')
  })

  it('should verify the page loads correctly', () => {
    // Check that the page title is correct
    cy.title().should('eq', 'PUEC')

    // Check that the main heading is visible
    cy.contains('PUEC Party').should('be.visible')

    // Check that the form exists
    cy.get('#filterForm').should('exist')

    // Check that instructor input exists
    cy.get('#instructor').should('exist')

    // Check that submit button exists
    cy.get('#submitBtn').should('exist')
  })
})
